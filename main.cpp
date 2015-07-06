#include "smo.h"
#include "para.h"
#include <iostream>
using namespace std;

int main()
{
    //

    Para para;

    para.Data = 10;
    para.TrainData = 8;
    para.In = 3;
    para.C = 1.0;
    para.Tor = 0.001;
    para.Eps = 1.0E-12;
    para.SigmaSquare = 2.0;

    SMO smo(para);
    smo.train("data/train.dat");
    smo.error_rate();
    smo.save();

    return 0;
}
