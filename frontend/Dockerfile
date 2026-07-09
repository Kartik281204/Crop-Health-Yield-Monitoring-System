# Crop Disease Detection -- frontend image.
#
# NOTE: syntax/logic-reviewed carefully but not build-tested -- no Docker
# in this sandbox. `npm run build` with this exact next.config.ts *has*
# been run and verified (see README); this Dockerfile just wraps that
# proven build in the standard 3-stage Next.js pattern.

# ---- deps: install once, cached separately from source changes ----
FROM node:22-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---- builder ----
FROM node:22-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_* vars are embedded into the client JS bundle at *build*
# time, not read at container runtime -- override with
# `docker build --build-arg NEXT_PUBLIC_API_URL=...` if the API isn't at
# localhost:8000 (e.g. deploying somewhere the browser reaches the API
# by a different host).
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# ---- runner: just the standalone server + static assets, no node_modules ----
FROM node:22-slim AS runner
WORKDIR /app
ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

CMD ["node", "server.js"]
