/*
 *   Copyright (c) 2023 Nicola Piccinelli
 *   All rights reserved.
 */
#include "basic.hpp"

int main() {
    constexpr int Tnx = 12;
    constexpr int Tny = 12;
    constexpr int Tnu = 4;
    constexpr int Tndu = 4;
    constexpr int Tph = 10;
    constexpr int Tch = 10;

#ifdef MPC_DYNAMIC
    mpc::LMPC<> optsolver(
        Tnx, Tnu, Tndu, Tny,
        Tph, Tch);
#else
    mpc::LMPC<
    TVAR(Tnx), TVAR(Tnu), TVAR(Tndu), TVAR(Tny),
            TVAR(Tph), TVAR(Tch)>
                       optsolver;
#endif

    optsolver.setLoggerLevel(mpc::Logger::log_level::NONE);

    mpc::mat<Tnx, Tnx> Ad;
    Ad << 1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0.1, 0, 0, 0,
            0.0488, 0, 0, 1, 0, 0, 0.0016, 0, 0, 0.0992, 0, 0,
            0, -0.0488, 0, 0, 1, 0, 0, -0.0016, 0, 0, 0.0992, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.0992,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0.9734, 0, 0, 0, 0, 0, 0.0488, 0, 0, 0.9846, 0, 0,
            0, -0.9734, 0, 0, 0, 0, 0, -0.0488, 0, 0, 0.9846, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9846;

    mpc::mat<Tnx, Tnu> Bd;
    Bd << 0, -0.0726, 0, 0.0726,
            -0.0726, 0, 0.0726, 0,
            -0.0152, 0.0152, -0.0152, 0.0152,
            0, -0.0006, -0.0000, 0.0006,
            0.0006, 0, -0.0006, 0,
            0.0106, 0.0106, 0.0106, 0.0106,
            0, -1.4512, 0, 1.4512,
            -1.4512, 0, 1.4512, 0,
            -0.3049, 0.3049, -0.3049, 0.3049,
            0, -0.0236, 0, 0.0236,
            0.0236, 0, -0.0236, 0,
            0.2107, 0.2107, 0.2107, 0.2107;

    mpc::mat<Tny, Tnx> Cd;
    Cd.setIdentity();

    mpc::mat<Tny, Tnu> Dd;
    Dd.setZero();

    optsolver.setStateSpaceModel(Ad, Bd, Cd);

    optsolver.setDisturbances(
            mpc::mat<Tnx, Tndu>::Zero(),
            mpc::mat<Tny, Tndu>::Zero());

    mpc::mat<Tnu, Tph> InputWMat, DeltaInputWMat;
    mpc::mat<Tny, Tph> OutputWMat;

    optsolver.setObjectiveWeights(OutputWMat, InputWMat, DeltaInputWMat);

    mpc::cvec<Tnu> InputW, DeltaInputW;
    mpc::cvec<Tny> OutputW;

    OutputW << 0, 0, 10, 10, 10, 10, 0, 0, 0, 5, 5, 5;
    InputW << 0.1, 0.1, 0.1, 0.1;
    DeltaInputW << 0, 0, 0, 0;

    optsolver.setObjectiveWeights(OutputW, InputW, DeltaInputW, {0, Tph});

    mpc::mat<Tnx, Tph> xminmat, xmaxmat;
    mpc::mat<Tny, Tph> yminmat, ymaxmat;
    mpc::mat<Tnu, Tph> uminmat, umaxmat;

    xminmat.setZero();
    xmaxmat.setZero();
    yminmat.setZero();
    ymaxmat.setZero();
    uminmat.setZero();
    umaxmat.setZero();

    optsolver.setConstraints(xminmat, uminmat, yminmat, xmaxmat, umaxmat, ymaxmat);

    mpc::cvec<Tnx> xmin, xmax;
    xmin << -M_PI / 6, -M_PI / 6, -mpc::inf, -mpc::inf, -mpc::inf, -1,
            -mpc::inf, -mpc::inf, -mpc::inf, -mpc::inf, -mpc::inf, -mpc::inf;

    xmax << M_PI / 6, M_PI / 6, mpc::inf, mpc::inf, mpc::inf, mpc::inf,
            mpc::inf, mpc::inf, mpc::inf, mpc::inf, mpc::inf, mpc::inf;

    mpc::cvec<Tny> ymin, ymax;
    ymin.setOnes();
    ymin *= -mpc::inf;
    ymax.setOnes();
    ymax *= mpc::inf;

    mpc::cvec<Tnu> umin, umax;
    double u0 = 10.5916;
    umin << 9.6, 9.6, 9.6, 9.6;
    umin.array() -= u0;
    umax << 13, 13, 13, 13;
    umax.array() -= u0;

    optsolver.setConstraints(xmin, umin, ymin, xmax, umax, ymax, {0, Tph});
    optsolver.setConstraints(xmin, umin, ymin, xmax, umax, ymax, {0, 1});

    optsolver.setScalarConstraint(-mpc::inf, mpc::inf, mpc::cvec<Tnx>::Ones(), mpc::cvec<Tnu>::Ones(), {-1, -1});
    optsolver.setScalarConstraint(0, -mpc::inf, mpc::inf, mpc::cvec<Tnx>::Ones(), mpc::cvec<Tnu>::Ones());

    optsolver.setReferences(mpc::mat<Tny, Tph>::Zero(), mpc::mat<Tnu, Tph>::Zero(), mpc::mat<Tnu, Tph>::Zero());

    mpc::cvec<Tny> yRef;
    yRef << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    optsolver.setReferences(yRef, mpc::cvec<Tnu>::Zero(), mpc::cvec<Tnu>::Zero(), {0, Tph});

    mpc::LParameters params;
    params.maximum_iteration = 250;
    optsolver.setOptimizerParameters(params);

    optsolver.setExogenuosInputs(mpc::mat<Tndu, Tph>::Zero());
    optsolver.setExogenuosInputs(mpc::cvec<Tndu>::Zero(), {0, Tph});

    auto res = optsolver.step(mpc::cvec<Tnx>::Zero(), mpc::cvec<Tnu>::Zero());
    auto seq = optsolver.getOptimalSequence();
    (void)seq;

    mpc::cvec<4> testRes;
    testRes << -0.9916, 1.74839, -0.9916, 1.74839;

    // Print the results
    std::cout << "Optimal Command: " << res.cmd.transpose() << std::endl;

    return 0;
}
