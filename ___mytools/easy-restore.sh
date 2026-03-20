#!/usr/bin/env bash
###############################################################################
# easy-restore.sh
# Sicheres Restore-Script für lokale Entwicklungsverzeichnisse
#
# Stellt ein zuvor mit easy-backup.sh erstelltes Archiv wieder her.
# ÜBERSCHREIBT alle vorhandenen Dateien im App-Verzeichnis!
#
# Sicherheitsfeatures:
#   - set -euo pipefail (strikte Fehlerbehandlung)
#   - Lock-File gegen parallele Ausführung
#   - SHA256-Integritätsprüfung vor dem Restore
#   - Doppelte Bestätigung mit Timeout
#   - Zip-Slip-Schutz (keine Pfade außerhalb des Zielverzeichnisses)
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

# ─── Farben ─────────────────────────────────────────────────────────────────
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

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
if ! command -v unzip &>/dev/null; then
    log ERROR "'unzip' ist nicht installiert. Bitte installieren."
    exit 1
fi

# SHA256-Abstraktion
sha256() {
    if command -v sha256sum &>/dev/null; then
        sha256sum "$@"
    else
        shasum -a 256 "$@"
    fi
}

# ─── Lock-File ───────────────────────────────────────────────────────────────
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

# ─── Archiv-Verzeichnis prüfen ──────────────────────────────────────────────
if [ ! -d "${ARCHIVE_DIR}" ]; then
    log ERROR "Archiv-Verzeichnis nicht gefunden: ${ARCHIVE_DIR}"
    log ERROR "Bitte zuerst ein Backup mit easy-backup.sh erstellen."
    exit 1
fi

# ─── Verfügbare Archive auflisten ───────────────────────────────────────────
archives=()
while IFS= read -r line; do
    [ -n "${line}" ] && archives+=("${line}")
done < <(
    find "${ARCHIVE_DIR}" -maxdepth 1 -name "easy-restore-${APP_NAME}-*.zip" -type f \
    | sort -r
)

if [ ${#archives[@]} -eq 0 ]; then
    log ERROR "Keine Archive für '${APP_NAME}' im Verzeichnis ${ARCHIVE_DIR} gefunden."
    exit 1
fi

# ─── Dynamische Box-Funktion ─────────────────────────────────────────────────
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
    "${GREEN}easy-restore${NC} – Lokales Entwicklungsarchiv wiederherstellen" \
    "" \
    "App-Verzeichnis : ${YELLOW}${APP_DIR}${NC}"
echo ""
echo -e "${CYAN}Verfügbare Archive:${NC}"
echo ""

for i in "${!archives[@]}"; do
    local_file="$(basename "${archives[$i]}")"
    local_size="$(du -h "${archives[$i]}" | cut -f1)"
    local_date="$(stat -c '%Y' "${archives[$i]}" 2>/dev/null || stat -f '%m' "${archives[$i]}" 2>/dev/null || echo "?")"

    # Datum menschenlesbar (Cross-Platform)
    if [[ "${local_date}" =~ ^[0-9]+$ ]]; then
        if date --version &>/dev/null 2>&1; then
            local_date="$(date -d "@${local_date}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "?")"
        else
            local_date="$(date -r "${local_date}" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "?")"
        fi
    fi

    printf "  ${GREEN}[%d]${NC}  %-55s  ${YELLOW}%s${NC}  %s\n" \
        $((i + 1)) "${local_file}" "${local_size}" "${local_date}"
done

echo ""

# ─── Archiv auswählen ────────────────────────────────────────────────────────
if [ ${#archives[@]} -eq 1 ]; then
    selected=0
    log INFO "Nur ein Archiv verfügbar – wird automatisch ausgewählt."
else
    while true; do
        read -r -p "Archiv-Nummer auswählen [1-${#archives[@]}] (0 = Abbruch): " choice
        if [ "${choice}" = "0" ]; then
            log INFO "Restore abgebrochen durch Benutzer."
            exit 0
        fi
        if [[ "${choice}" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#archives[@]} )); then
            selected=$((choice - 1))
            break
        fi
        echo -e "${RED}Ungültige Eingabe. Bitte eine Zahl zwischen 1 und ${#archives[@]} eingeben.${NC}"
    done
fi

readonly ARCHIVE_PATH="${archives[$selected]}"
readonly ARCHIVE_NAME="$(basename "${ARCHIVE_PATH}" .zip)"
readonly CHECKSUM_PATH="${ARCHIVE_DIR}/${ARCHIVE_NAME}.sha256"

echo ""
log INFO "Gewähltes Archiv: ${ARCHIVE_NAME}.zip"

# ─── Integritätsprüfung (SHA256) ─────────────────────────────────────────────
if [ -f "${CHECKSUM_PATH}" ]; then
    log INFO "Prüfe Integrität (SHA256) ..."
    expected_hash="$(cut -d' ' -f1 "${CHECKSUM_PATH}")"
    actual_hash="$(cd "${ARCHIVE_DIR}" && sha256 "$(basename "${ARCHIVE_PATH}")" | cut -d' ' -f1)"

    if [ "${expected_hash}" != "${actual_hash}" ]; then
        echo ""
        log ERROR "╔══════════════════════════════════════════════════════════╗"
        log ERROR "║  INTEGRITÄTSFEHLER! Das Archiv wurde verändert!         ║"
        log ERROR "║  Erwartet : ${expected_hash:0:32}...  ║"
        log ERROR "║  Ist      : ${actual_hash:0:32}...  ║"
        log ERROR "╚══════════════════════════════════════════════════════════╝"
        echo ""
        log ERROR "Restore wird aus Sicherheitsgründen ABGEBROCHEN."
        exit 1
    fi
    log OK "Integritätsprüfung bestanden."
else
    log WARN "Keine Checksumme vorhanden (${ARCHIVE_NAME}.sha256 fehlt)."
    log WARN "Integritätsprüfung wird übersprungen."
fi

# ─── Zip-Slip-Prüfung ───────────────────────────────────────────────────────
log INFO "Prüfe Archiv-Inhalt auf verdächtige Pfade ..."
suspicious=0
while IFS= read -r entry; do
    # Prüfe auf Pfad-Traversal (../ oder absolute Pfade)
    if [[ "${entry}" == /* ]] || [[ "${entry}" == *../* ]] || [[ "${entry}" == */../* ]]; then
        log ERROR "VERDÄCHTIGER PFAD im Archiv: ${entry}"
        suspicious=$((suspicious + 1))
    fi
done < <(unzip -l "${ARCHIVE_PATH}" 2>/dev/null | awk 'NR>3 {print $4}' | sed '$d' | sed '$d')

if (( suspicious > 0 )); then
    log ERROR "Archiv enthält ${suspicious} verdächtige Pfade! Restore wird ABGEBROCHEN."
    exit 1
fi
log OK "Archiv-Inhalt geprüft – keine verdächtigen Pfade."

# ─── WARNUNG & Doppelte Bestätigung ─────────────────────────────────────────
echo ""
print_box "${RED}" \
    "WARNUNG: ALLE DATEIEN WERDEN ÜBERSCHRIEBEN!" \
    "" \
    "Zielverzeichnis: ${APP_DIR}" \
    "" \
    "Alle vorhandenen Dateien (ausser ${TOOLS_DIR_NAME}/) werden" \
    "unwiderruflich mit dem Archivinhalt ersetzt!"

# Erste Bestätigung
read -r -t 60 -p "Restore wirklich durchführen? (ja/nein): " confirm1 || true
confirm1_lower="$(echo "${confirm1}" | tr '[:upper:]' '[:lower:]')"
if [ "${confirm1_lower}" != "ja" ]; then
    log INFO "Restore abgebrochen (1. Bestätigung)."
    echo -e "${YELLOW}Restore abgebrochen.${NC}"
    exit 0
fi

# Zweite Bestätigung (Sicherheit)
echo ""
echo -e "${RED}Letzte Chance! Eingabe 'RESTORE' zur endgültigen Bestätigung:${NC}"
read -r -t 30 -p "> " confirm2 || true
if [ "${confirm2}" != "RESTORE" ]; then
    log INFO "Restore abgebrochen (2. Bestätigung)."
    echo -e "${YELLOW}Restore abgebrochen.${NC}"
    exit 0
fi

# ─── Restore durchführen ─────────────────────────────────────────────────────
echo ""
log INFO "Starte Restore von ${ARCHIVE_NAME}.zip ..."

cd "${APP_DIR}"

# Entpacken mit Überschreiben, ___mytools/ wird ausgelassen
unzip -o "${ARCHIVE_PATH}" -x "${TOOLS_DIR_NAME}/*" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo ""
    log OK "Restore erfolgreich abgeschlossen!"
    echo ""
    echo -e "  Quelle : ${GREEN}${ARCHIVE_NAME}.zip${NC}"
    echo -e "  Ziel   : ${GREEN}${APP_DIR}${NC}"
    echo ""
    log INFO "Restore abgeschlossen: ${ARCHIVE_NAME}.zip → ${APP_DIR}"
else
    log ERROR "Restore fehlgeschlagen! Prüfe die Logdatei: ${LOG_FILE}"
    exit 1
fi