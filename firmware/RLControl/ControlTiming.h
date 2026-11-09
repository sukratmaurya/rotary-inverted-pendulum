#pragma once

#include <stdint.h>

// Fixed-rate gate for the 50 Hz trained controller. The controller's
// observation/action contract and its discrete 1/f control math are defined
// at the nominal period; this helper never stretches dt to wall-clock time.
namespace ControlTiming
{
enum Decision : uint8_t
{
    WAIT_FOR_DEADLINE,
    RUN_ONE_TICK,
    REJECT_LATE_TICK
};

inline Decision decide(uint32_t elapsed_us, uint32_t period_us,
                       uint32_t allowed_lateness_us)
{
    if (elapsed_us < period_us) return WAIT_FOR_DEADLINE;
    if (elapsed_us - period_us > allowed_lateness_us) return REJECT_LATE_TICK;
    return RUN_ONE_TICK;
}

// Unsigned subtraction intentionally supports micros() rollover.
inline bool timestamp_is_fresh(uint32_t now_us, uint32_t sample_us,
                               uint32_t max_age_us)
{
    return static_cast<uint32_t>(now_us - sample_us) <= max_age_us;
}

class FixedRateScheduler
{
public:
    void begin(uint32_t now_us) { last_due_us_ = now_us; }

    // Consume at most one due slot and rebase to its observation time. A
    // rejected late slot also rebases, so resuming the caller cannot replay
    // missed slots as a burst.
    Decision poll(uint32_t now_us, uint32_t period_us,
                  uint32_t allowed_lateness_us)
    {
        Decision result = decide(static_cast<uint32_t>(now_us - last_due_us_),
                                 period_us, allowed_lateness_us);
        if (result != WAIT_FOR_DEADLINE) last_due_us_ = now_us;
        return result;
    }

private:
    uint32_t last_due_us_ = 0;
};
}  // namespace ControlTiming
