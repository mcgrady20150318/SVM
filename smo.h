#ifndef SMO_H
#define SMO_H

#include "para.h"
#include <vector>
using namespace std;

#define MAX(a,b)  ((a)>(b)?(a):(b))
#define MIN(a,b)  ((a)<(b)?(a):(b))

class SMO{

public:
    SMO(Para);
    ~SMO();

    void train(const char *inputDataPath); //训练分类器
    void save();  //将分类器保存下来
    void error_rate();   //计算分类正确率
    int predict(double *);   //对单个数据进行预测

private:
    bool takeStep( int i1, int i2 );   //优化两个拉格朗日乘子
    double ui(int i1);  //分类输出，对应公式10
    double kernelRBF(int, int );  //径向基核函数
    double kernelRBF(int, double* ); //径向基核函数的重载，用于prediction
    double dotProduct(int i1,int i2);   //两个训练样本的点积
    int examineExample(int );
    void  readFile(const char *);
    int examineFirstChoice(int i1,double E1);
    int examineNonBound(int );
    int examineBound(int );
    void outerLoop();   //论文中的外层循环，在伪代码中是在主函数部分


private:
    int Data;				//所有的样本数
    int TrainData;				//训练的样本数
    int In;		    	 //数据的维数
    double C;				   //惩罚系数
    double Tor;			    	//在KKT条件中容忍范围
    double Eps;                  //限制条件
    double SigmaSquare;     //RBF核函数中的参数
    double b;                          //阈值
    vector <int> Target;           //类别标签
    double **data;                  //存放训练与测试样本
    vector<double> alpha;               //拉格朗日乘子
    vector<double> ErrorCache;         //存放non-bound样本误差
    vector<double> DotProductCache;    //预存向量的点积以减少计算量

};

#endif
