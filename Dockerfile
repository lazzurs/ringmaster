# syntax=docker/dockerfile:1

# Multi-arch build without emulation: the builder always runs on the build
# host's platform and cross-compiles for TARGETPLATFORM with tonistiigi/xx.
# The runtime stage only copies files, so no RUN executes under QEMU.

FROM --platform=$BUILDPLATFORM tonistiigi/xx:1.9.0@sha256:c64defb9ed5a91eacb37f96ccc3d4cd72521c4bd18d5442905b95e2226b0e707 AS xx

FROM --platform=$BUILDPLATFORM rust:1.98-trixie@sha256:a8a5f0a1e5fe7dfe1d352591e4a1c7dd2c08fd70475cae872cf3458ba0df0546 AS build
COPY --from=xx / /
RUN apt-get update \
 && apt-get install -y --no-install-recommends clang lld \
 && rm -rf /var/lib/apt/lists/*
ARG TARGETPLATFORM
# Target sysroot: libc and libgcc for linking, a C compiler for SQLite and aws-lc.
RUN xx-apt-get install -y --no-install-recommends gcc g++ libc6-dev
WORKDIR /src

# Dependency layer: build all dependencies against stub sources so this layer
# is reused until Cargo.toml or Cargo.lock change.
COPY Cargo.toml Cargo.lock ./
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    mkdir -p src \
 && echo 'fn main() {}' > src/main.rs \
 && echo '' > src/lib.rs \
 && xx-cargo build --release --locked \
 && rm -rf src

COPY src ./src
ARG RINGMASTER_VERSION
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    touch src/main.rs src/lib.rs \
 && RINGMASTER_VERSION="${RINGMASTER_VERSION}" xx-cargo build --release --locked \
 && cp "target/$(xx-cargo --print-target-triple)/release/ringmaster" /ringmaster \
 && xx-verify /ringmaster \
 && mkdir -p /data-skel /config-skel

# distroless/cc: glibc + libgcc + CA certificates, no shell, no package manager.
FROM gcr.io/distroless/cc-debian13:nonroot@sha256:532c7a94f9e96ba9db75eb9388f4227751131c7cb0a411d1a8521af4096135de
ARG RINGMASTER_VERSION
LABEL org.opencontainers.image.title="ringmaster" \
      org.opencontainers.image.description="Voice commands from a smart ring: webhook, local transcription, Claude agent with MCP tools" \
      org.opencontainers.image.source="https://github.com/lazzurs/ringmaster" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      org.opencontainers.image.version="${RINGMASTER_VERSION}"
COPY --from=build /ringmaster /usr/local/bin/ringmaster
# /data is the only writable path (SQLite + recordings); owned by nonroot.
COPY --from=build --chown=65532:65532 /data-skel /data
COPY --from=build /config-skel /etc/ringmaster
USER 65532:65532
ENV RINGMASTER_CONFIG=/etc/ringmaster/config.toml
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["/usr/local/bin/ringmaster", "healthcheck"]
ENTRYPOINT ["/usr/local/bin/ringmaster"]
CMD ["serve"]
