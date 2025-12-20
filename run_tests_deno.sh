#!/bin/bash -e


./deno.sh test                  \
    --allow-read=.,/tmp         \
    --no-prompt                 \
    --cached-only               \
    ${@-tests/}


