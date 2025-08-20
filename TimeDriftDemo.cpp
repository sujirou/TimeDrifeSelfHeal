// single-file C++17 example of a tiny "time discipline" engine that:
//  - estimates and corrects clock drift when offline using a frequency estimate (ppm)
//  - smoothly slews toward NTP/authoritative time when samples arrive
//  - persists the learned drift to disk so restarts keep improving
//
// NOTE: this is user-space demo code. real OSes discipline the kernel clock and RTC.
//       here we just maintain a model of "what time should be" and expose it via now().
//
// Build:  g++ -std=gnu++17 -O2 -pthread TimeDriftDemo.cpp -o timedrift
// Run:    ./timedrift

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

using namespace std::chrono;

// tiny helper to format time_points
static std::string fmt_system_time(system_clock::time_point tp) {
    std::time_t t = system_clock::to_time_t(tp);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    auto ms = duration_cast<milliseconds>(tp.time_since_epoch()) % 1000;
    char out[80];
    std::snprintf(out, sizeof(out), "%s.%03lld", buf, (long long)ms.count());
    return out;
}

// TimeSyncEngine maintains a disciplined clock estimate.
class TimeSyncEngine {
public:
    // Max slew rate (how fast we correct offset) in parts per million.
    // 500 ppm is similar magnitude to Unix adjtime/chrony defaults.
    static constexpr double MAX_SLEW_PPM = 500.0; 

    // persistent state file for drift (ppm) and last good offset.
    explicit TimeSyncEngine(std::string state_path = "time_drift.cal")
        : state_path_(std::move(state_path)) {
        load_state();
        // initialize base anchors to "now"
        base_wall_ = system_clock::now();
        base_mono_ = steady_clock::now();
        last_update_mono_ = base_mono_;
    }

    // called when an authoritative time sample (e.g., NTP) arrives.
    // ntp_time: the true current time (system_clock domain) from server
    // recv_wall: our local notion of system_clock at the moment we received the sample
    void on_ntp_sample(system_clock::time_point ntp_time,
                       system_clock::time_point recv_wall) {
        std::lock_guard<std::mutex> lock(mu_);
        auto now_mono = steady_clock::now();

        // compute observed offset: how far our model's time is from true time at recv.
        auto model_at_recv = model_time_unlocked(now_mono);
        auto offset = duration_cast<nanoseconds>(ntp_time - model_at_recv);

        // update frequency estimate if we have a previous sample (Allan-like estimate)
        if (last_ntp_true_.has_value()) {
            auto dt_true = duration<double>(ntp_time - *last_ntp_true_).count();
            auto dt_mono = duration<double>(now_mono - last_ntp_mono_).count();
            if (dt_true > 0.5 && dt_mono > 0.5) {
                // freq error ppm = (true_rate - mono_rate) / mono_rate * 1e6
                // treat mono as ideal 1.0; any difference becomes frequency correction.
                double ppm = ((dt_true / dt_mono) - 1.0) * 1e6;
                // exponentially weighted moving average to smooth noise
                const double alpha = 0.2; // tuneable
                freq_ppm_ = (1.0 - alpha) * freq_ppm_ + alpha * ppm;
            }
        }

        // set a new target offset to be slewed toward over time.
        target_offset_ = offset;

        // re-anchor base points so future model_time uses updated state smoothly
        reanchor_unlocked(now_mono);

        last_ntp_true_ = ntp_time;
        last_ntp_mono_ = now_mono;
        last_sync_wall_ = recv_wall;
        save_state();
    }

    // returns the current disciplined time estimate.
    system_clock::time_point now() {
        std::lock_guard<std::mutex> lock(mu_);
        return model_time_unlocked(steady_clock::now());
    }

    // simulate monotonic tick to advance slewing.
    void tick() {
        std::lock_guard<std::mutex> lock(mu_);
        (void)model_time_unlocked(steady_clock::now()); // advancing applies slew
    }

    // for demos: get current frequency estimate (ppm) and pending offset (ns)
    double freq_ppm() const {
        std::lock_guard<std::mutex> lock(mu_);
        return freq_ppm_;
    }
    long long pending_offset_ns() const {
        std::lock_guard<std::mutex> lock(mu_);
        return target_offset_.count();
    }

private:
    // core model: time = base_wall + elapsed * (1 + freq) + applied_slew(elapsed)
    system_clock::time_point model_time_unlocked(steady_clock::time_point mono_now) {
        auto elapsed = duration<double>(mono_now - base_mono_).count();

        // apply frequency correction
        double scale = 1.0 + freq_ppm_ * 1e-6;
        auto freq_term = duration_cast<nanoseconds>(duration<double>(elapsed * scale));

        // apply bounded slewing toward target_offset_
        auto dt = duration<double>(mono_now - last_update_mono_).count();
        last_update_mono_ = mono_now;

        // how much offset can we correct over this small step at MAX_SLEW_PPM?
        // convert ppm limit into ns per second of real time
        double max_correction_per_sec = MAX_SLEW_PPM * 1e-6; // fraction per sec
        // slew acts on current time value, approximate as linear over dt
        auto max_ns = (long long)std::llround((max_correction_per_sec * dt) * 1e9);

        long long to_apply = 0;
        if (target_offset_.count() != 0) {
            long long want = target_offset_.count();
            if (std::llabs(want) <= std::llabs(max_ns)) {
                to_apply = want;
            } else {
                to_apply = (want > 0) ? max_ns : -max_ns;
            }
            // reduce remaining target by what we applied
            target_offset_ -= nanoseconds(to_apply);
            applied_slew_ += nanoseconds(to_apply);
        }

        // Time = base + elapsed*scale + applied_slew
        return base_wall_ + freq_term + applied_slew_;
    }

    void reanchor_unlocked(steady_clock::time_point mono_now) {
        // move base_wall_ so that model_time remains continuous at mono_now
        auto current_model = base_wall_ + duration_cast<nanoseconds>(mono_now - base_mono_)
                             + applied_slew_;
        base_wall_ = current_model; // anchors at current model time
        base_mono_ = mono_now;
        last_update_mono_ = mono_now;
        applied_slew_ = nanoseconds(0); // incorporate it into the new base
    }

    void load_state() {
        std::ifstream f(state_path_);
        if (!f) return;
        double ppm = 0.0;
        long long offset_ns = 0;
        if (f >> ppm >> offset_ns) {
            freq_ppm_ = ppm;
            target_offset_ = nanoseconds(offset_ns);
        }
    }

    void save_state() {
        std::ofstream f(state_path_, std::ios::trunc);
        if (!f) return;
        f << std::setprecision(12) << freq_ppm_ << "\n" << target_offset_.count() << "\n";
    }

    // STATE madaflekah!
    std::string state_path_;

    mutable std::mutex mu_;
    system_clock::time_point base_wall_{};     // anchor in wall-time domain
    steady_clock::time_point base_mono_{};     // anchor in monotonic domain
    steady_clock::time_point last_update_mono_{};

    // frequency error estimate (ppm). Positive means our clock runs too fast.
    double freq_ppm_ = 0.0;

    // slew management
    nanoseconds target_offset_{0}; // remaining offset to correct toward true time
    nanoseconds applied_slew_{0};  // accumulated since last (re)anchor

    // bookkeeping of last authoritative sample
    std::optional<system_clock::time_point> last_ntp_true_;
    steady_clock::time_point last_ntp_mono_{};
    std::optional<system_clock::time_point> last_sync_wall_;
};

// --- Demo harness -----------------------------------------------------------
// simulate a local clock that runs a bit fast, and periodic NTP samples.

int main() {
    TimeSyncEngine engine;

    std::cout << "Starting demo. Learned freq_ppm (initial): " << engine.freq_ppm() << "\n";

    // simulate for ~30 seconds. Every 7 seconds we get an NTP sample.
    auto start_true = system_clock::now();
    auto start_mono = steady_clock::now();

    // let's pretend the true time advances normally, but our host clock runs +120 ppm fast.
    const double host_ppm_bias = 120.0; // typical quartz drift magnitude

    for (int i = 0; i < 30; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        // advance our simulated host "recv_wall" by bias (fast clock)
        auto mono_now = steady_clock::now();
        double dt = duration<double>(mono_now - start_mono).count();
        auto recv_wall = start_true + duration_cast<nanoseconds>(duration<double>(dt * (1.0 + host_ppm_bias*1e-6)));

        // occasionally deliver an NTP sample (perfect true time)
        if (i % 7 == 0 && i != 0) {
            auto ntp_true = start_true + duration_cast<nanoseconds>(duration<double>(dt));
            engine.on_ntp_sample(ntp_true, recv_wall);
            std::cout << "[NTP] sample @ t=" << i << "s  freq_ppm=" << engine.freq_ppm()
                      << "  pending_offset(ns)=" << engine.pending_offset_ns() << "\n";
        }

        // engine ticks (applies slewing)
        engine.tick();

        // report the disciplined time vs true
        auto est = engine.now();
        auto true_now = start_true + duration_cast<nanoseconds>(duration<double>(dt));
        auto err_ms = duration_cast<milliseconds>(est - true_now).count();

        std::cout << "t=" << std::setw(2) << i+1 << "s  est=" << fmt_system_time(est)
                  << "  true=" << fmt_system_time(true_now)
                  << "  error=" << err_ms << " ms\n";
    }

    std::cout << "\nDone. Final learned freq_ppm: " << engine.freq_ppm() << " (should trend toward ~" << host_ppm_bias << ")\n";
}
