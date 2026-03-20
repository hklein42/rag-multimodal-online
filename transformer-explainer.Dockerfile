FROM nginx:alpine
COPY transformer-explainer/build /usr/share/nginx/html
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    add_header Cross-Origin-Opener-Policy "same-origin";\n\
    add_header Cross-Origin-Embedder-Policy "credentialless";\n\
    add_header Cross-Origin-Resource-Policy "cross-origin";\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf
EXPOSE 80
