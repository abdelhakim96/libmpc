

#include <mpc/NLMPC.hpp>
#include <iostream>
using namespace std;

constexpr bool DYNAMIC_ALLOC = false;

constexpr int TVAR(int v)
{
    return v;
}

int main()
{


    constexpr int num_states = 8;
    constexpr int num_output = 8;
    constexpr int num_inputs = 4;
    constexpr int pred_hor = 5;
    constexpr int ctrl_hor = 5;
    constexpr int ineq_c = pred_hor + 1;
    constexpr int eq_c = 0;

    double ts = 0.001;


    Eigen::Matrix<double, 8, 8> Q; // State weight matrix
    Eigen::Matrix<double, 4, 4> R; // Control input weight matrix
    Eigen::Matrix<double, 8, 1> X_r; // State reference matrix

// Define the state and control input weights
    Q.setZero(); // Initialize Q as a matrix of zeros
    R.setZero(); // Initialize R as an identity matrix
    X_r.setZero(); // Initialize X_R as a matrix of zeros

// You can adjust the weights based on your control objectives
    Q(0, 0) = 50.0;   // Weight x
    Q(1, 1) = 50.0;   // Weight y
    Q(2, 2) = 2.0;   // Weight z
    Q(3, 3) = 0.1;
    Q(4, 4) = 0.1;
    Q(5, 5) = 0.0;
    Q(6, 6) = 0.0;
    Q(7, 7) = 0.0;   // Weight for state 8

    R(0, 0) = 0.1;   // Weight for control input 1
    R(1, 1) = 0.1;   // Weight for control input 2
    R(2, 2) = 0.1;   // Weight for control input 3
    R(3, 3) = 0.1;   // Weight for control input 4


    X_r(0, 0) = 1.0;   // reference x
    X_r(1, 0) = 3.0;   // reference x
    X_r(2, 0) = 0.0;   // reference y
    X_r(3, 0) = 0.0;
    X_r(4, 0) = 0.0;
    X_r(5, 0) = 0.0;
    X_r(6, 0) = 0.0;
    X_r(7, 0) = 0.0;   // Weight for state 8

    mpc::NLMPC<
        num_states, num_inputs, num_output,
        pred_hor, ctrl_hor,
        ineq_c, eq_c>
        optsolver;


    optsolver.setContinuosTimeModel(ts);




    //ROV Model params
    // BlueROV2 Model Parameters
    const double m = 11.5;    // BlueROV2 mass (kg)
    const double g = 9.81;  // gravitional field strength (m/s^2)
    const double F_bouy = 114.8; // Bouyancy force (N)

    const double X_ud = -5.5 ; // Added mass in x direction (kg)
    const double Y_vd = -12.7 ; // Added mass in y direction (kg)
    const double Z_wd = -14.57 ; // Added mass in z direction (kg)
    const double N_rd = -0.12  ; // Added mass for rotation about z direction (kg)

    const double I_xx = 0.16 ; // Moment of inertia (kg.m^2)
    const double I_yy = 0.16 ; // Moment of inertia (kg.m^2)
    const double I_zz = 0.16 ; // Moment of inertia (kg.m^2)

    const double X_u = -4.03 ; // Linear damping coefficient in x direction (N.s/m)
    const double Y_v  = -6.22 ; // Linear damping coefficient  in y direction (N.s/m)
    const double Z_w = -5.18 ; // Linear damping coefficient  in z direction (N.s/m)
    const double N_r = -0.07 ;  // Linear damping coefficient for rotation about z direction (N.s/rad)

    const double X_uc = -18.18 ; // quadratic damping coefficient in x direction (N.s^2/m^2)
    const double Y_vc = -21.66 ; // quadratic damping coefficient  in y direction (N.s^2/m^2)
    const double Z_wc = -36.99 ; // quadratic damping coefficient  in z direction (N.s^2/m^2)
    const double N_rc = -1.55  ; // quadratic damping coefficient for rotation about z direction (N.s^2/rad^2)
    const double eps = 0.001;  // offset to prevent numerical issues of square root



    auto stateEq = [&](
                       mpc::cvec<TVAR(num_states)> &dx,
                       const mpc::cvec<TVAR(num_states)>& x,
                       const mpc::cvec<TVAR(num_inputs)>& u)
    {
        //x(0) = x
        //x(1) = y
        //x(2) = z
        //x(3) = u
        //x(4) = v
        //x(5) = w
        //x(6) = psi
        //x(7) = r

        dx(0)= cos(x(6)) * x(3) - sin(x(6)) * x(4);
        dx(1) =  sin(x(6)) * x(3) +  cos(x(6)) * x(4);
        dx(2) = x(5);

        dx(3) = (u(0) + (m * x(4) + Y_vd * x(4)) * x(7) + (X_u + X_uc *sqrt( x(3) * x(3) + eps) ) * x(3))/(m - X_ud)  ;
        dx(4) = (u(1) - (m * x(3) + X_ud * x(3)) * x(7) + (Y_v + Y_vc *sqrt( x(4) * x(4) + eps) ) * x(4))/(m - Y_vd) ;
        dx(5)= (u(2) + (Z_w + Z_wc * sqrt(x(5) * x(5) + eps)) * x(5) + (m * g - F_bouy))/(m - Z_wd) ;

        dx(6) = x(7);
        dx(7) = (u(3) - (m * x(4) - Y_vd * x(4)) * x(3) - (X_ud * x(3) - m * x(3)) * x(4) + (N_r + N_rc * sqrt(x(7) * x(7) + eps)) * x(7))/(I_zz - N_rd);

        //dx(0) = ((1.0 - (x(1) * x(1))) * x(0)) - x(1) + u(0);
        //dx(1) = x(0);
    };

    optsolver.setStateSpaceFunction([&](
                                        mpc::cvec<TVAR(num_states)> &dx,
                                        const mpc::cvec<TVAR(num_states)>& x,
                                        const mpc::cvec<TVAR(num_inputs)>& u,
                                        const unsigned int&)
                                    { stateEq(dx, x, u); });

    optsolver.setObjectiveFunction([&](
                                       const mpc::mat<TVAR(pred_hor + 1), TVAR(num_states)> &x,
                                       const mpc::mat<TVAR(pred_hor + 1), TVAR(num_output)> &,
                                       const mpc::mat<TVAR(pred_hor + 1), TVAR(num_inputs)> &u,
                                       double)
                                   //{

                                  //     mpc::mat<TVAR(pred_hor + 1), TVAR(num_states)> cost;

                                  {
                                      mpc::mat<1, TVAR(num_states)> xi;
                                for (int i ; i<x.rows(); i++ ) {
                                   xi = (x.block<1, TVAR(num_states)>(i, 0)-X_r.transpose());
                                }
                                  double cost = xi * Q * xi.transpose();

                                  return cost;});
                                  //return   x.array().square().sum() ; });


    optsolver.setIneqConFunction([&](
                                     mpc::cvec<TVAR(ineq_c)> &in_con,
                                     const mpc::mat<TVAR(pred_hor + 1), TVAR(num_states)>&,
                                     const mpc::mat<TVAR(pred_hor + 1), TVAR(num_output)>&,
                                     const mpc::mat<TVAR(pred_hor + 1), TVAR(num_inputs)>& u,
                                     const double&)
                                 {
       // for (int i = 0; i < ineq_c; i++) {
        //    in_con(i) = u(i, 0) - 0.5;
        //}
                                     in_con(0) = u(0, 0) - 1.0;
                                     in_con(1) = u(1, 0) - 1.0;
                                     in_con(2) = u(3, 0) - 1.0;
                                     in_con(3) = u(4, 0) - 0.01;


                                 });



    mpc::cvec<TVAR(num_states)> modelX, modeldX;
    modelX.resize(num_states);
    modeldX.resize(num_states);

    modelX(0) = 0;
    modelX(1) = 1.0;

    auto r = optsolver.getLastResult();

    for (;;)
    {
        r = optsolver.step(modelX, r.cmd);
        auto seq = optsolver.getOptimalSequence();
        (void) seq;
        stateEq(modeldX, modelX, r.cmd);
        modelX += modeldX * ts;

        cout <<"x position: " << modelX(0)<<endl ;
        cout <<"y position: " << modelX(1)<<endl;
        //cout <<"control: " << r.cmd<<endl;

        //if (std::fabs(modelX[0]) <= 1e-2 && std::fabs(modelX[1]) <= 1e-1)
        //{
        //    break;
        //}
    }

    return 0;
}

