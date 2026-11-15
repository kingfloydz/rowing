#pragma once
// SSPRK(4,3), Ketcheson 2008, DOI 10.1137/07070485X.
// Four Euler maps of duration dt/2 with one convex combination.
// RHS times are 0, 1/2, 1, 1/2; final time is exactly dt.
struct RkScheme {
    bool four_stage=false;
    int stages() const {return four_stage?4:3;}
    double euler_fraction() const {return four_stage?.5:1.;}
    double abscissa(int stage) const {
        return four_stage?(stage==0?0.:stage==2?1.:.5):(stage==0?0.:stage==1?1.:.5);
    }
    double base_weight(int stage) const {
        return four_stage?(stage==2?2./3:0.):(stage==0?0.:stage==1?.75:1./3);
    }
};
