/* orkd_2proc — genuine TWO-PROCESS orkd proof. fork()+exec()s N *separate* client binaries (default ./test_orkd)
 * that each connect to the one orkd and run concurrently, then asserts every child exits 0. Unlike the earlier
 * fork-and-run-in-child multi-tenant attempt (which wedged — a forked child inherited orkd/NPU state and then
 * auto-spawned), each child here is a FRESH exec'd image doing a normal orkd_connect, so it's the real
 * separate-process case. The daemon serializes their submits onto the single-stream NPU.
 *
 *   make orkd test_orkd orkd_2proc
 *   sudo env ORK_USE_ORKD=1 ORKD_BIN=$PWD/orkd ./orkd_2proc [child] [nproc]
 * (pre-start one orkd, or let the first child auto-spawn it; flock arbitrates the spawn race.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv){
    const char *child = argc > 1 ? argv[1] : "./test_orkd";
    int n = argc > 2 ? atoi(argv[2]) : 2; if (n < 1) n = 1; if (n > 16) n = 16;
    pid_t pid[16];
    fprintf(stderr, "[2proc] launching %d concurrent '%s' processes against one orkd\n", n, child);
    for (int i = 0; i < n; i++){
        pid[i] = fork();
        if (pid[i] < 0){ perror("fork"); return 2; }
        if (pid[i] == 0){ execl(child, child, (char *)NULL); perror("execl"); _exit(127); }   /* fresh image */
    }
    int fail = 0;
    for (int i = 0; i < n; i++){
        int st; waitpid(pid[i], &st, 0);
        int rc = WIFEXITED(st) ? WEXITSTATUS(st) : (WIFSIGNALED(st) ? 128 + WTERMSIG(st) : -1);
        fprintf(stderr, "[2proc] child %d (pid %d) exit=%d\n", i, (int)pid[i], rc);
        if (rc != 0) fail = 1;
    }
    printf("TWO_PROCESS_ORKD: %s — %d concurrent separate processes through one daemon\n", fail ? "FAIL" : "PASS", n);
    return fail ? 1 : 0;
}
