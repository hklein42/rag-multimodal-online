#!/usr/bin/env bash
###############################################################################
# easy-backup.sh
# Sicheres Backup-Script für lokale Entwicklungsverzeichnisse
#
# Legt ein geziptes Archiv aller Dateien des übergeordneten App-Verzeichnisses
# im Unterverzeichnis ___mytools/__archive/ ab.
# Das Verzeichnis ___mytools/ (inkl. aller Unterverzeichnisse) wird NICHT
# ins Backup aufgenommen.
#
# Namenskonvention:
#   easy-restore-<app-name>-<YYYYMMDD-HHMMSS>_v<NNN>.zip
#   (komplett kleingeschrieben)
#
# Sicherheitsfeatures:
#   - set -euo pipefail (strikte Fehlerbehandlung)
#   - Lock-File gegen parallele Ausführung
#   - SHA256-Checksumme für Integritätsprüfung
#   - Restriktive Dateirechte (600) auf Archiv + Checksumme
#   - Symlink-Warnung
#   - Logging aller Operationen
#
# Autor:  Heinzpeter (generiert mit Claude)
# Datum:  2026-03-19
###############################################################################
set -euo pipefail

# ─── Konfiguration ──────────────────────────────────────────────────────────
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly APP_NAME="$(basename "${APP_DIR}" | tr '[:upper:]' '[:lower:]')"
readonly ARCHIVE_DIR="${SCRIPT_DIR}/__archive"
readonly LOCK_FILE="${SCRIPT_DIR}/.easy-backup.lock"
readonly LOG_FILE="${ARCHIVE_DIR}/easy-backup.log"
readonly TOOLS_DIR_NAME="___mytools"
readonly TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

# ─── Farben ─────────────────────────────────────────────────────────────────
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# ─── Hilfsfunktionen ────────────────────────────────────────────────────────
log() {
    local level="$1"; shift
    local msg="$*"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${ts}] [${level}] ${msg}" >> "${LOG_FILE}" 2>/dev/null || true
    case "${level}" in
        INFO)  echo -e "${BLUE}[INFO]${NC}  ${msg}" ;;
        OK)    echo -e "${GREEN}[OK]${NC}    ${msg}" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC}  ${msg}" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} ${msg}" >&2 ;;
    esac
}

cleanup() {
    rm -f "${LOCK_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ─── Voraussetzungen prüfen ─────────────────────────────────────────────────
if ! command -v zip &>/dev/null; then
    log ERROR "'zip' ist nicht installiert. Bitte installieren (apt install zip / brew install zip)."
    exit 1
fi

if ! command -v sha256sum &>/dev/null && ! command -v shasum &>/dev/null; then
    log ERROR "Weder 'sha256sum' noch 'shasum' gefunden. Bitte installieren."
    exit 1
fi

# SHA256-Abstraktion (Linux: sha256sum, macOS: shasum -a 256)
sha256() {
    if command -v sha256sum &>/dev/null; then
        sha256sum "$@"
    else
        shasum -a 256 "$@"
    fi
}

# ─── Lock-File (Schutz gegen parallele Ausführung) ──────────────────────────
if [ -f "${LOCK_FILE}" ]; then
    lock_pid="$(cat "${LOCK_FILE}" 2>/dev/null || echo "")"
    if [ -n "${lock_pid}" ] && kill -0 "${lock_pid}" 2>/dev/null; then
        log ERROR "Ein anderer Backup/Restore-Prozess läuft bereits (PID: ${lock_pid})."
        exit 1
    else
        log WARN "Verwaistes Lock-File gefunden – wird entfernt."
        rm -f "${LOCK_FILE}"
    fi
fi
echo $$ > "${LOCK_FILE}"

# ─── Archiv-Verzeichnis anlegen ─────────────────────────────────────────────
mkdir -p "${ARCHIVE_DIR}"
chmod 700 "${ARCHIVE_DIR}"

# ─── Versionsnummer ermitteln ────────────────────────────────────────────────
# Sucht das höchste v<NNN> im Archiv-Verzeichnis für das heutige Datum-Präfix
# und zählt hoch. Falls keines existiert, beginnt bei v001.
get_next_version() {
    local prefix="easy-restore-${APP_NAME}-${TIMESTAMP%%-*}"  # bis zum Datum
    local max_ver=0

    # Alle bestehenden Archive durchsuchen
    for f in "${ARCHIVE_DIR}"/easy-restore-"${APP_NAME}"-*.zip; do
        [ -f "$f" ] || continue
        local fname
        fname="$(basename "$f" .zip)"
        # Versionsnummer extrahieren
        if [[ "${fname}" =~ _v([0-9]+)$ ]]; then
            local ver="${BASH_REMATCH[1]#0}"  # führende Nullen entfernen
            ver="${ver:-0}"
            if (( ver > max_ver )); then
                max_ver=$ver
            fi
        fi
    done

    printf "v%03d" $(( max_ver + 1 ))
}

VERSION="$(get_next_version)"
readonly ARCHIVE_NAME="easy-restore-${APP_NAME}-${TIMESTAMP}_${VERSION}"
readonly ARCHIVE_PATH="${ARCHIVE_DIR}/${ARCHIVE_NAME}.zip"
readonly CHECKSUM_PATH="${ARCHIVE_DIR}/${ARCHIVE_NAME}.sha256"

# ─── Symlink-Prüfung ────────────────────────────────────────────────────────
symlink_count=0
while IFS= read -r -d '' link; do
    symlink_count=$((symlink_count + 1))
    if [ "${symlink_count}" -le 10 ]; then
        log WARN "Symlink gefunden (wird NICHT ins Archiv aufgenommen): ${link}"
    fi
done < <(find "${APP_DIR}" -path "${SCRIPT_DIR}" -prune -o -type l -print0 2>/dev/null)

if (( symlink_count > 10 )); then
    log WARN "... und $(( symlink_count - 10 )) weitere Symlinks."
fi
if (( symlink_count > 0 )); then
    log WARN "Insgesamt ${symlink_count} Symlink(s) werden übersprungen (Sicherheit)."
fi

# ─── Info-Ausgabe ────────────────────────────────────────────────────────────
print_box() {
    local color="$1"; shift
    local lines=("$@")
    local max_len=0
    for line in "${lines[@]}"; do
        local plain
        plain="$(echo "$line" | sed 's/\x1b\[[0-9;]*m//g')"
        local len=${#plain}
        (( len > max_len )) && max_len=$len
    done
    local width=$(( max_len + 4 ))
    local border
    border="$(printf '═%.0s' $(seq 1 "$width"))"

    echo ""
    echo -e "${color}╔${border}╗${NC}"
    for line in "${lines[@]}"; do
        local plain
        plain="$(echo "$line" | sed 's/\x1b\[[0-9;]*m//g')"
        local pad=$(( width - ${#plain} - 2 ))
        local spaces=""
        if (( pad > 0 )); then
            spaces="$(printf ' %.0s' $(seq 1 "$pad"))"
        fi
        echo -e "${color}║${NC} ${line}${spaces} ${color}║${NC}"
    done
    echo -e "${color}╚${border}╝${NC}"
    echo ""
}

print_box "${BLUE}" \
    "${GREEN}easy-backup${NC} – Lokales Entwicklungsarchiv" \
    "" \
    "App-Verzeichnis : ${YELLOW}${APP_DIR}${NC}" \
    "Archiv-Name     : ${YELLOW}${ARCHIVE_NAME}.zip${NC}" \
    "Version         : ${YELLOW}${VERSION}${NC}" \
    "Ausschluss      : ${YELLOW}${TOOLS_DIR_NAME}/${NC} (komplett)"

# ─── Backup erstellen ────────────────────────────────────────────────────────
log INFO "Starte Backup ..."

cd "${APP_DIR}"

# zip erstellen: ___mytools/ komplett ausschließen, keine Symlinks folgen
zip -r -y \
    "${ARCHIVE_PATH}" \
    . \
    -x "./${TOOLS_DIR_NAME}/*" \
    -x "./${TOOLS_DIR_NAME}" \
    > /dev/null 2>&1

if [ ! -f "${ARCHIVE_PATH}" ]; then
    log ERROR "Archiv konnte nicht erstellt werden!"
    exit 1
fi

# ─── Rechte setzen ───────────────────────────────────────────────────────────
chmod 600 "${ARCHIVE_PATH}"

# ─── Checksumme erstellen ────────────────────────────────────────────────────
(cd "${ARCHIVE_DIR}" && sha256 "$(basename "${ARCHIVE_PATH}")") > "${CHECKSUM_PATH}"
chmod 600 "${CHECKSUM_PATH}"

# ─── Statistiken ─────────────────────────────────────────────────────────────
archive_size="$(du -h "${ARCHIVE_PATH}" | cut -f1)"
file_count="$(unzip -l "${ARCHIVE_PATH}" 2>/dev/null | tail -1 | awk '{print $2}')"
checksum_val="$(cut -d' ' -f1 "${CHECKSUM_PATH}")"

echo ""
log OK "Backup erfolgreich erstellt!"
echo ""
echo -e "  Archiv   : ${GREEN}${ARCHIVE_PATH}${NC}"
echo -e "  Größe    : ${GREEN}${archive_size}${NC}"
echo -e "  Dateien  : ${GREEN}${file_count}${NC}"
echo -e "  SHA256   : ${GREEN}${checksum_val}${NC}"
echo ""

log INFO "Backup abgeschlossen: ${ARCHIVE_NAME}.zip (${archive_size}, ${file_count} Dateien)"