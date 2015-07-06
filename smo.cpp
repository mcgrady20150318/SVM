#include "smo.h"
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <ctime>
using namespace std;

//初始化SMO，除了使用参数结构体以外，还要将各个
//vector预设大小，提高程序的运行效率
SMO::SMO(Para para)
{
    Data = para.Data;
    TrainData = para.TrainData;

    In = para.In;

    C = para.C;
    Tor = para.Tor;
    Eps = para.Eps;
    SigmaSquare = para.SigmaSquare;

    b = 0.0;
    //初始化二维数组
    data = new double*[Data];
    for ( int i = 0; i < Data; ++i )
        data[i] = new double[In];

    for ( int i = 0; i < Data; ++i )
        for (int j = 0; j < In; ++j )
            data[i][j] = 0.0;

    Target.resize(Data,0);
    alpha.resize(TrainData,0);
    ErrorCache.resize(TrainData,0);
    DotProductCache.resize(Data,0);

}

//输出类别，对应公式10
double SMO::ui(int k)
{
    int i;
    double s = 0;
    for(i = 0; i < TrainData; i++)
        if(alpha[i] > 0)
            s += alpha[i] * Target[i] * kernelRBF(i,k);
    s -= b;
    return s;
}

int SMO::predict(double *array)
{
    double s = 0;
    for(int i = 0; i < TrainData; i++)
        if(alpha[i] > 0)
            s += alpha[i] * Target[i] * kernelRBF(i,array);
    s -= b;
    return s>0?1:-1;
}

//点积，求两个点的内积
double SMO::dotProduct(int i1,int i2)
{
    double dot = 0;
    for(int i = 0; i < In; i++)
        dot += data[i1][i] * data[i2][i];
    return dot;
}

//径向基核函数，这个核函数使用的是高斯核函数
double SMO::kernelRBF(int i1,int i2 )
{
    double s = dotProduct(i1,i2);
    s *= -2;
    s += DotProductCache[i1] + DotProductCache[i2];
    return exp(-s / SigmaSquare);
}

//重载径向基核函数，被predict函数调用
double SMO::kernelRBF(int i1,double *inputData)
{
    double s = 0;
    double dDotProiputData = 0;  //输入数据的点积
    double dDoti1 = 0;  //i1的点积
    for ( int i = 0; i < In; ++i )
    {
       dDoti1 += data[i1][i] * data[i1][i];
       dDotProiputData += inputData[i] * inputData[i];
       s += data[i1][i] * inputData[i];
    }

    s *= -2;
    s += dDoti1 + dDotProiputData;
    return exp(-s / SigmaSquare);

}

//优化两个拉格朗日系数，参考论文中伪代码
bool SMO::takeStep(int i1,int i2 )
{
    if ( i1 == i2 )
        return 0;

    double s  = 0;
    double E1 = 0,E2 = 0;
    double L  = 0, H = 0;
    double k11,k12,k22;
    double eta = 0;
    double a1,a2;
    double Lobj,Hobj;

    double alph1 = alpha[i1];
    double alph2 = alpha[i2];
    double y1 = Target[i1];
    double y2 = Target[i2];

    if( ErrorCache[i1] > 0 && ErrorCache[i1] < C )
        E1 = ErrorCache[i1];
    else
        E1 = ui(i1) - y1;

    s = y1 * y2;

    //Compute L, H via equations (13) and (14)
    if( y1 == y2 )
    {
        L = MAX(alph2 + alph1 - C , 0 );
        H = MIN(alph1 + alph2 , C );
    }
    else
    {
        L = MAX(alph2 - alph1 , 0 );
        H = MIN(C , C + alph1 + alph2 );
    }

    if ( L == H )
        return 0;

    k11 = kernelRBF(i1,i1);
    k12 = kernelRBF(i1,i2);
    k22 = kernelRBF(i2,i2);

    eta = k11 + k22 - 2*k12;

    if (eta > 0)
    {
        a2 = alph2 + y2 * (E1-E2)/eta;
        if(a2 < L)
            a2 = L;
        else if( a2 > H )
            a2 = H;
    }
    else
    {
        double f1 = y1*(E1 + b) - alph1*k11 - s*alph2*k12;
        double f2 = y2*(E2 + b) - s*alph1*k12 - alph2*k22;
        double L1 = alph1 + s*(alph2 - L);
        double H1 = alph1 + s*(alph2 - H);
        Lobj = L1*f1 + L*f2 + (L1*L1*k11 + L*L*k22)/2 + s*L*L1*k12;
        Hobj = H1*f1 + H*f2 + (H1*H1*k11 + H*H*k22)/2 + s*H*H1*k12;

        if ( Lobj < Hobj - Eps )
            a2 = L;
        else if ( Lobj > Hobj + Eps )
            a2 = H;
        else
            a2 = alph2;
    }

    if ( abs(a2-alph2) < Eps*(a2+alph2+Eps))
        return 0;

    a1 = alph1 + s*(alph2 - a2);

    //Update threshold to reflect change in Lagrange multipliers
    double b1 = E1 + y1*(a1-alph1)*k11 + y2*(a2-alph2)*k12 + b;
    double b2 = E2 + y1*(a1-alph1)*k12 + y2*(a2-alph2)*k22 + b;
    double delta_b = b;
    b = (b1 + b2) / 2.0;
    delta_b = b - delta_b;

    //Update error cache using new Lagrange multipliers
    double t1 = y1 * (a1 - alph1);
    double t2 = y2 * (a2 - alph2);
    for(int i = 0; i < TrainData; i++)
        if(alpha[i] > 0 && alpha[i] < C)
            ErrorCache[i] += t1 * kernelRBF(i1,i) + t2 * (kernelRBF(i2,i)) - delta_b;
    ErrorCache[i1] = 0;
    ErrorCache[i2] = 0;

    //Store a1,a2 in the alpha array
    alpha[i1] = a1;
    alpha[i2] = a2;

    return 1;

}





//使用启发式的方法，实现inner loop来选择第二个乘子
//这个函数被outer loop调用
int SMO::examineExample(int i1)
{
    double y1 = Target[i1];
    double alph1 = alpha[i1];
    double E1;

    if( ErrorCache[i1] > 0 && ErrorCache[i1] < C )
        E1 = ErrorCache[i1];
    else
        E1 = ui(i1) - y1;

    double r1 = E1 * y1;

    if ( (r1 < - Tor && alph1 < C ) || ( r1 > Tor && alph1 > 0))
    {
    /*
        使用三种方法选择第二个乘子
        1：在non-bound乘子中寻找maximum fabs(E1-E2)的样本
        2：如果上面没取得进展,那么从随机位置查找non-boundary 样本
        3：如果上面也失败，则从随机位置查找整个样本,改为bound样本
        */
        if(examineFirstChoice(i1,E1))  return 1;  //第1种情况
        if(examineNonBound(i1))  return 1;  //第2种情况
        if(examineBound(i1))  return 1;  //第3种情况
    }
    return 0;

}


//1：在non-bound乘子中寻找maximum fabs(E1-E2)的样本
int SMO::examineFirstChoice(int i1,double E1)
{
    int k,i2;
    double tmax;
    double E2,temp;
    for(i2 = - 1,tmax = 0,k = 0; k < TrainData; k++)
    {
        if(alpha[k] > 0 && alpha[k] < C)
        {
            E2 = ErrorCache[k];
            temp = fabs(E1 - E2);
            if(temp > tmax)
            {
                tmax = temp;
                i2 = k;
            }
        }
    }
    if(i2 >= 0 && takeStep(i1,i2))  return 1;
    return 0;
}

//	2：如果上面没取得进展,那么从随机位置查找non-boundary样本
int SMO::examineNonBound(int i1)
{
    int k0 = rand() % TrainData;
    int k,i2;
    for(k = 0; k < TrainData; k++)
    {
        i2 = (k + k0) % TrainData;
        if((alpha[i2] > 0 && alpha[i2] < C) && takeStep(i1,i2))  return 1;
    }
    return 0;
}

//  3：如果上面也失败，则从随机位置查找整个样本,(改为bound样本)
int SMO::examineBound(int i1)
{
    int k0 = rand() % TrainData;
    int k,i2;

    for(k = 0; k < TrainData; k++)
    {
        i2 = (k + k0) % TrainData;
        if(takeStep(i1,i2))  return 1;
    }
    return 0;
}


/**********************************
inputDatapath：训练数据保存的路径
s： SMO的参数结构体
***********************************/
void SMO::train(const char *inputDataPath)
{

    readFile(inputDataPath);

    //设置预计算点积（对训练样本的设置，对于测试样本也要考虑）
    for(int i = 0; i < Data; i++)
        DotProductCache[i] = dotProduct(i,i);

    outerLoop();

    int count = 0;

    for(int i=0;i<TrainData;i++){

        if(alpha[i] > 0){

            cout << alpha[i] << endl;

            count ++;
        }


    }

    cout << count << " support vectors " << endl;

}

//这里使用的是libsvm中的经典的heart_scal数据集中的格式
void SMO::readFile(const char* filePath){

    int i,j;

    ifstream infile(filePath);

    string line;

    for(i=0;i<Data;i++){

        getline(infile, line);

        istringstream stream(line);

        string field;

        for(j=0;j<In+1;j++){

            getline(stream,field,' ');

            if(j < In){

                data[i][j] = atof(field.c_str());

            }else{


                Target[i] = (atoi(field.c_str()));

            }

        }

    }

    infile.close();

}


//计算分类误差率
void SMO::error_rate()
{
    int ac = 0;
    double accuracy,tar;

    for(int i = TrainData; i < Data; i++)
    {
        tar = ui(i);
        if(tar > 0 && Target[i] > 0 || tar < 0 && Target[i] < 0)   ac++;
        cout<<"th test value is  "<<tar<<endl;
    }
    accuracy = (double)ac / (Data - TrainData);
    cout<<"精确度："<<accuracy * 100<<"％"<<endl;
}



//对应论文中的outer loop，用来寻找第一个要优化的乘子
void SMO::outerLoop()
{
    int numChanged = 0;
    bool examineAll = 1;

    while ( numChanged > 0 || examineAll )
    {
        numChanged = 0;
        if ( examineAll )
        {
            for ( int i = 0; i < TrainData; ++i )
                numChanged += examineExample(i);
        }
        else
        {
            for ( int i = 0; i < TrainData; ++i )
            {
                if (alpha[i] > 0 && alpha[i] < C )
                    numChanged += examineExample(i);
            }
        }
        if ( examineAll == 1 )
            examineAll = 0;
        else if ( numChanged == 0 )
            examineAll = 1;

    }

}

//将支持向量及相关必要的信息保存下来
void SMO::save()
{
    ofstream outfile("svm.txt");

    int countVec = 0;   //支持向量的个数
    for ( int i = 0; i < TrainData; ++i )
        if ( alpha[i] > 0 )
            ++countVec;

    //第一行保存支持向量的个数，数据的维数，还有高斯核参数,阈值b
    outfile<<countVec<<' '<<In<<' '<<SigmaSquare<<' '<<b<<'\n';
    for ( int i = 0; i < TrainData; ++i )
    {
        if ( alpha[i] > 0 )
        {
            outfile<<Target[i]<<' '<<alpha[i]<<' ';
            for ( int j = 0; j < In; ++j )
                outfile<<data[i][j]<<' ';
            outfile<<'\n';
        }
    }
}

//将保存的训练结果（即分类器）加载进来，用于分类


SMO::~SMO()
{
    for (int i = 0; i < Data; ++i )
    {
        delete [] data[i];
    }
    delete []data;
}
