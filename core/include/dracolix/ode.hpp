#pragma once
// ODE solvers — Phase 6: Euler, RK4, Adaptive RK45 (Dormand-Prince simplified)
// Licensed under GPL-3.0-only
#include "array.hpp"
#include <functional>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace dracolix::ode {

using State = std::vector<double>;
using RHS = std::function<State(double t, const State& y)>;

inline State add_state(const State& a, const State& b, double s=1.0){
    State r(a.size());
    for(size_t i=0;i<a.size();++i) r[i]=a[i]+s*b[i];
    return r;
}
inline State scale_state(const State& a, double s){
    State r(a.size());
    for(size_t i=0;i<a.size();++i) r[i]=a[i]*s;
    return r;
}

// Fixed-step Euler
inline std::vector<State> euler(RHS f, double t0, State y0, double t1, double dt){
    if(dt<=0) throw std::invalid_argument("dt>0");
    std::vector<State> traj; traj.reserve(size_t((t1-t0)/dt)+1);
    double t=t0; State y=y0;
    traj.push_back(y);
    while(t < t1 - 1e-12){
        double h = std::min(dt, t1-t);
        State dy = f(t,y);
        for(size_t i=0;i<y.size();++i) y[i] += h*dy[i];
        t+=h; traj.push_back(y);
    }
    return traj;
}

// Classical RK4
inline std::vector<State> rk4(RHS f, double t0, State y0, double t1, double dt){
    if(dt<=0) throw std::invalid_argument("dt>0");
    std::vector<State> traj; traj.reserve(size_t((t1-t0)/dt)+1);
    double t=t0; State y=y0; traj.push_back(y);
    while(t < t1 - 1e-12){
        double h = std::min(dt, t1-t);
        State k1=f(t,y);
        State k2=f(t+h/2, add_state(y, scale_state(k1, h/2)));
        State k3=f(t+h/2, add_state(y, scale_state(k2, h/2)));
        State k4=f(t+h, add_state(y, scale_state(k3, h)));
        for(size_t i=0;i<y.size();++i) y[i] += h/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]);
        t+=h; traj.push_back(y);
    }
    return traj;
}

// Adaptive RK45 with simple step doubling (embedded error estimate)
inline std::vector<State> adaptive_rk45(RHS f, double t0, State y0, double t1, double tol=1e-6, double h0=0.1){
    if(h0<=0) throw std::invalid_argument("h0>0");
    std::vector<State> traj; traj.push_back(y0);
    double t=t0; State y=y0; double h=h0;
    auto rk_step = [&](double tt, const State& yy, double hh){
        State k1=f(tt,yy);
        State k2=f(tt+hh/2, add_state(yy, scale_state(k1, hh/2)));
        State k3=f(tt+hh/2, add_state(yy, scale_state(k2, hh/2)));
        State k4=f(tt+hh, add_state(yy, scale_state(k3, hh)));
        State y4(yy.size()); for(size_t i=0;i<yy.size();++i) y4[i]=yy[i]+hh/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]);
        // half steps for error
        State y_half = yy;
        double th=tt;
        for(int s=0;s<2;++s){
            double hs=hh/2;
            State k1h=f(th,y_half);
            State k2h=f(th+hs/2, add_state(y_half, scale_state(k1h, hs/2)));
            State k3h=f(th+hs/2, add_state(y_half, scale_state(k2h, hs/2)));
            State k4h=f(th+hs, add_state(y_half, scale_state(k3h, hs)));
            for(size_t i=0;i<y_half.size();++i) y_half[i]+=hs/6*(k1h[i]+2*k2h[i]+2*k3h[i]+k4h[i]);
            th+=hs;
        }
        return std::pair<State,State>(y4,y_half);
    };
    while(t < t1 - 1e-12){
        if(t+h>t1) h=t1-t;
        auto [y4, y_half]= rk_step(t,y,h);
        double err=0; for(size_t i=0;i<y.size();++i) err+=std::pow(y4[i]-y_half[i],2); err=std::sqrt(err);
        if(err < tol || h < 1e-12){
            y=y_half; t+=h; traj.push_back(y);
            if(err < tol/10) h*=1.5;
        } else {
            h*=0.5;
        }
        if(traj.size()>100000) break;
    }
    return traj;
}

} // namespace dracolix::ode
