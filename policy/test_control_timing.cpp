#include <assert.h>
#include <stdint.h>

#include "../firmware/RLControl/ControlTiming.h"

int main()
{
    constexpr uint32_t period = 20000;
    constexpr uint32_t tolerance = 1000;

    // Normal fixed-rate cadence and the pre-deadline wait state.
    assert(ControlTiming::decide(period - 1, period, tolerance) ==
           ControlTiming::WAIT_FOR_DEADLINE);
    assert(ControlTiming::decide(period, period, tolerance) ==
           ControlTiming::RUN_ONE_TICK);
    assert(ControlTiming::decide(period + tolerance, period, tolerance) ==
           ControlTiming::RUN_ONE_TICK);

    // Bounded jitter gets exactly one update. There is no elapsed-dt scaling
    // and no API state that can emit multiple catch-up ticks.
    const auto jittered = ControlTiming::decide(period + 500, period, tolerance);
    assert(jittered == ControlTiming::RUN_ONE_TICK);

    // Beyond the documented tolerance, including one or more missed periods,
    // the caller must fault and skip policy/controller execution.
    assert(ControlTiming::decide(period + tolerance + 1, period, tolerance) ==
           ControlTiming::REJECT_LATE_TICK);
    assert(ControlTiming::decide(2 * period, period, tolerance) ==
           ControlTiming::REJECT_LATE_TICK);
    assert(ControlTiming::decide(5 * period, period, tolerance) ==
           ControlTiming::REJECT_LATE_TICK);

    // Stateful schedule: consume one normal slot, accept a bounded delayed
    // slot once, then prove repeated polling cannot replay/catch up that slot.
    ControlTiming::FixedRateScheduler scheduler;
    scheduler.begin(100000);
    assert(scheduler.poll(119999, period, tolerance) ==
           ControlTiming::WAIT_FOR_DEADLINE);
    assert(scheduler.poll(120000, period, tolerance) ==
           ControlTiming::RUN_ONE_TICK);
    assert(scheduler.poll(140500, period, tolerance) ==
           ControlTiming::RUN_ONE_TICK);
    assert(scheduler.poll(140500, period, tolerance) ==
           ControlTiming::WAIT_FOR_DEADLINE);
    assert(scheduler.poll(160500, period, tolerance) ==
           ControlTiming::RUN_ONE_TICK);

    // A missed slot is rejected and consumed; it is never emitted again as
    // a burst on the next poll.
    assert(scheduler.poll(201501, period, tolerance) ==
           ControlTiming::REJECT_LATE_TICK);
    assert(scheduler.poll(201501, period, tolerance) ==
           ControlTiming::WAIT_FOR_DEADLINE);

    // Sample-age validation handles exact boundary, stale values, and the
    // unsigned micros() wraparound case.
    assert(ControlTiming::timestamp_is_fresh(15000, 5000, 10000));
    assert(!ControlTiming::timestamp_is_fresh(15001, 5000, 10000));
    assert(ControlTiming::timestamp_is_fresh(5, UINT32_MAX - 4, 10));
    assert(!ControlTiming::timestamp_is_fresh(20, UINT32_MAX - 4, 10));
    return 0;
}
