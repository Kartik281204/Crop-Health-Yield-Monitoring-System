import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Produces a minimal, self-contained server bundle (.next/standalone) so
  // the Docker image doesn't need to ship full node_modules -- see
  // frontend/Dockerfile.
  output: "standalone",
};

export default nextConfig;
