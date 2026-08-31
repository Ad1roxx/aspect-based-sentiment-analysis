# ---------------------------------------------------------------------------
# ABSA frontend — React built to static files, served by nginx.
#
#     docker build -t absa-web --build-arg VITE_API_URL=http://localhost:8000 .
#
# THE BUILD ARG IS LOAD-BEARING. Vite inlines import.meta.env at BUILD time, so
# the API URL is baked into the JavaScript and cannot be changed by setting an
# environment variable on the running container.
#
# It must be an address the BROWSER can reach. The browser runs on the host,
# outside the compose network, so it has no idea what the hostname "api" means —
# http://api:8000 would resolve inside the network and fail in the page. Use
# http://localhost:8000. This builds and starts cleanly either way and only
# breaks when someone clicks the button, which is what makes it worth a comment.
# ---------------------------------------------------------------------------

FROM node:20-alpine AS builder
WORKDIR /app

# package.json and the lockfile first, so `npm ci` is only re-run when
# dependencies actually change rather than on every source edit.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build


# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
