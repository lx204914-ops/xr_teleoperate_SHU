#include <ctime>
#include <cstdint>
extern "C" void cbf_wait_until(const double deadline) {
    const std::int64_t target=static_cast<std::int64_t>(deadline*1000000000.0);
    timespec now{};
    do { clock_gettime(CLOCK_MONOTONIC,&now); }
    while(static_cast<std::int64_t>(now.tv_sec)*1000000000LL+now.tv_nsec<target);
}
