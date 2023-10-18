#include "basic.hpp"

int main() {
    constexpr int Tnx = 12;
    constexpr int Tny = 12;
    constexpr int Tnu = 4;
    constexpr int Tndu = 4;
    constexpr int Tph = 10;
    constexpr int Tch = 10;

    mpc::LMPC<> optsolver(Tnx, Tnu, Tndu, Tny, Tph, Tch);

    optsolver.setLoggerLevel(mpc::Logger::log_level::NONE);

    // ... (rest of the code, setting up the MPC solver and parameters)

    auto res = optsolver.step(mpc::cvec<Tnx>::Zero(), mpc::cvec<Tnu>::Zero());
    auto seq = optsolver.getOptimalSequence();

    // ... (perform other actions with the MPC solver)

    return 0;
}
