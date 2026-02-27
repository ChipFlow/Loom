# CVC (open-src-cvc) Verilog simulator for SDF timing validation.
# Build: docker build --platform linux/amd64 -t loom-cvc tests/timing_test/cvc
#
# CVC compiles Verilog to native machine code, so the runtime image
# needs gcc and binutils in addition to the cvc64 binary.
ARG PLATFORM=linux/amd64
FROM --platform=${PLATFORM} debian:bookworm-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git ca-certificates zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/cambridgehackers/open-src-cvc.git /cvc-src && \
    cd /cvc-src/src && \
    make -f makefile.cvc64 -j"$(nproc)" && \
    cp cvc64 /usr/local/bin/cvc64

FROM --platform=${PLATFORM} debian:bookworm-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc binutils zlib1g zlib1g-dev libc6-dev && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/bin/cvc64 /usr/local/bin/cvc64
WORKDIR /work
ENTRYPOINT ["cvc64"]
