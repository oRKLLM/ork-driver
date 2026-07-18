/* orkd_probe — validate the orkd daemon lifecycle end-to-end (board tool, NOT in `make test`).
 *
 * Auto-spawns orkd if none is running (the daemon opens the NPU), connects as a subscriber, prints the
 * daemon-reported NPU core count, does a PING/PONG round-trip, then disconnects. With no args it runs one
 * client; `orkd_probe <n>` holds the connection <n> seconds (run two to watch them SHARE one daemon +
 * idle-reap after both leave). Env: ORKD_BIN (daemon path), ORKD_IDLE_MS, XDG_RUNTIME_DIR (endpoint dir).
 *
 * Run under sudo so the auto-spawned daemon can open /dev/dri/cardN:
 *   make orkd orkd_probe && sudo env ORKD_BIN=$PWD/orkd ./orkd_probe
 */
#include "orkd_client.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv){
    int hold = argc > 1 ? atoi(argv[1]) : 0;
    orkd_conn *c = orkd_connect();
    if (!c){ fprintf(stderr, "orkd_probe: connect/spawn FAILED\n"); return 1; }
    printf("[pid %d] connected: client_id=%u npu_cores=%u\n", (int)getpid(), orkd_client_id(c), orkd_soc_cores(c));
    int prc = orkd_ping(c);
    printf("[pid %d] ping %s\n", (int)getpid(), prc == 0 ? "OK" : "FAILED");
    if (hold > 0) sleep(hold);
    orkd_disconnect(c);
    printf("[pid %d] disconnected\n", (int)getpid());
    return prc == 0 ? 0 : 2;
}
