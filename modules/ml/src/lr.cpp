///////////////////////////////////////////////////////////////////////////////////////
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.

// This is a implementation of the Logistic Regression algorithm in C++ in OpenCV.

// AUTHOR:
// Rahul Kavi rahulkavi[at]live[at]com

// # You are free to use, change, or redistribute the code in any way you wish for
// # non-commercial purposes, but please maintain the name of the original author.
// # This code comes with no warranty of any kind.

// #
// # You are free to use, change, or redistribute the code in any way you wish for
// # non-commercial purposes, but please maintain the name of the original author.
// # This code comes with no warranty of any kind.

// # Logistic Regression ALGORITHM


//                           License Agreement
//                For Open Source Computer Vision Library

// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:

//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.

//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.

//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.

// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.

#include "precomp.hpp"

using namespace std;

namespace cv {
namespace ml {

class LrParams
{
public:
    LrParams()
    {
        alpha = 0.001;
        num_iters = 1000;
        norm = LogisticRegression::REG_L2;
        train_method = LogisticRegression::BATCH;
        mini_batch_size = 1;
        term_crit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, num_iters, alpha);
        test_data_is_validation = false;
        max_iters_no_improvement = 20;
        record_train_freq = 100;
        train_parallel = true;
        decrease_alpha = true;
    }

    double alpha; //!< learning rate.
    int num_iters; //!< number of iterations.
    int norm;
    int train_method;
    int mini_batch_size;
    TermCriteria term_crit;
    bool test_data_is_validation; //!< true to use the test portion of the input training data as a validation set
    int max_iters_no_improvement; //!< max number of iterations to keep training if there's no improvement on validation set accuracy
    bool record_training; //!< true to record the cost and validation accuracy (if computed) for each training iteration
    int record_train_freq;
    bool train_parallel;
    bool decrease_alpha; // true to decrease the learning rate if no progress is being made relative to the validation set
};

class LogisticRegressionImpl : public LogisticRegression
{
public:

    LogisticRegressionImpl() { }
    virtual ~LogisticRegressionImpl() {}

    CV_IMPL_PROPERTY(double, LearningRate, params.alpha)
    CV_IMPL_PROPERTY(int, Iterations, params.num_iters)
    CV_IMPL_PROPERTY(int, Regularization, params.norm)
    CV_IMPL_PROPERTY(int, TrainMethod, params.train_method)
    CV_IMPL_PROPERTY(int, MiniBatchSize, params.mini_batch_size)
    CV_IMPL_PROPERTY(TermCriteria, TermCriteria, params.term_crit)
    CV_IMPL_PROPERTY(bool, UseValidationData, params.test_data_is_validation )
    CV_IMPL_PROPERTY(int, MaxItersNoValidImprovement, params.max_iters_no_improvement )
    CV_IMPL_PROPERTY(bool, RecordTrainingPerf, params.record_training )

    virtual bool train( const Ptr<TrainData>& trainData, int=0 );
    virtual float predict(InputArray samples, OutputArray results, int flags=0) const;
    virtual void clear();
    virtual void write(FileStorage& fs) const;
    virtual void read(const FileNode& fn);
    virtual Mat get_learnt_thetas() const { return learnt_thetas; }
    virtual Mat2f getTrainingPerf() const { return training_perf; }
    virtual int getVarCount() const { return learnt_thetas.cols; }
    virtual bool isTrained() const { return !learnt_thetas.empty(); }
    virtual bool isClassifier() const { return true; }
    virtual String getDefaultName() const { return "opencv_ml_lr"; }
protected:
//    void train_multiclass( const Mat & data_t,
//            const Mat & labels_l, const Mat & _data_val, const Mat & _labels_val,
//            const Mat & thetas, const Mat & init_theta, bool parallelize );
    float compute_prediction(InputArray samples, const Mat& _thetas, OutputArray results, int flags ) const;
    Mat calc_sigmoid(const Mat& data) const;
    double compute_cost(const Mat& _data, const Mat& _labels, const Mat& _init_theta);
    void compute_gradient(const Mat& _data, const Mat& _labels, const Mat &_theta, const double _lambda, Mat & _gradient );
    Mat batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta, const Mat& _data_val, const Mat& _labels_val, Mat2f perf);
    Mat mini_batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta, const Mat& _data_val, const Mat& _labels_val, Mat2f perf);
    bool set_label_map(const Mat& _labels_i);
    void remap_labels(Mat& labels_i, const map<int, int>& lmap) const;
protected:
    LrParams params;
    Mat learnt_thetas;
    Mat2f training_perf;
    map<int, int> forward_mapper;
    map<int, int> reverse_mapper;
    Mat labels_o;
    Mat labels_n;
};

Ptr<LogisticRegression> LogisticRegression::create()
{
    return makePtr<LogisticRegressionImpl>();
}

bool LogisticRegressionImpl::train(const Ptr<TrainData>& trainData, int)
{
    // return value
    bool ok = false;

    clear();
    Mat _data_i = trainData->getTrainSamples();
    Mat _labels_i = trainData->getTrainResponses();

    Mat _data_val, _labels_val;
    if ( params.test_data_is_validation &&
            trainData->getNTestSamples() > 0 &&
            params.train_method == LogisticRegression::MINI_BATCH &&
            params.max_iters_no_improvement > 0 )
    {
        _data_val = trainData->getTestSamples();
        _labels_val = trainData->getTestResponses();
    }

    // check size and type of training data
    CV_Assert( !_labels_i.empty() && !_data_i.empty());
    if(_labels_i.cols != 1)
    {
        CV_Error( CV_StsBadArg, "labels should be a column matrix" );
    }
    if(_data_i.type() != CV_32FC1 || _labels_i.type() != CV_32FC1)
    {
        CV_Error( CV_StsBadArg, "data and labels must be a floating point matrix" );
    }
    if(_labels_i.rows != _data_i.rows)
    {
        CV_Error( CV_StsBadArg, "number of rows in data and labels should be equal" );
    }

    // class labels
    set_label_map(_labels_i);
    Mat labels_l;
    _labels_i.convertTo(labels_l, CV_32S);
    remap_labels(labels_l, this->forward_mapper);

    int num_classes = (int) this->forward_mapper.size();
    if(num_classes < 2)
    {
        CV_Error( CV_StsBadArg, "data should have at least 2 classes" );
    }

    int num_perf_iters = max(params.num_iters, params.term_crit.maxCount) / params.record_train_freq;
    training_perf = Mat2f::zeros(2 + num_perf_iters, num_classes == 2 ? 1 : num_classes);

    // add a column of ones to the data (bias/intercept term)
    Mat data_t;
    hconcat( cv::Mat::ones( _data_i.rows, 1, CV_32F ), _data_i, data_t );

    // coefficient matrix (zero-initialized)
    Mat thetas;
    Mat init_theta = Mat::zeros(data_t.cols, 1, CV_32F);

    Mat new_theta;
    Mat labels, labels_val;

    // fit the model (handles binary and multiclass cases)
    if(num_classes == 2)
    {
        labels_l.convertTo(labels, CV_32F);
        if(this->params.train_method == LogisticRegression::BATCH)
            new_theta = batch_gradient_descent(data_t, labels, init_theta, _data_val, _labels_val, training_perf);
        else
            new_theta = mini_batch_gradient_descent(data_t, labels, init_theta, _data_val, _labels_val, training_perf);
        thetas = new_theta.t();
    }
    else
    {
        /* take each class and rename classes you will get a theta per class
        as in multi class class scenario, we will have n thetas for n classes */
        thetas.create(num_classes, data_t.cols, CV_32F);

//        Mat new_theta;
        Mat labels_binary, labels_bin_val;
//        Mat labels, labels_val;
        int ii = 0;
        for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
        {
            // one-vs-rest (OvR) scheme
            labels_binary = (labels_l == it->second)/255;
            labels_binary.convertTo(labels, CV_32F);

            labels_bin_val = (_labels_val == it->second)/255;
            labels_bin_val.convertTo(labels_val, CV_32S);

            if( params.train_method == BATCH)
                new_theta = batch_gradient_descent(data_t, labels, init_theta, _data_val, labels_val, training_perf.col(ii));
            else
                new_theta = mini_batch_gradient_descent(data_t, labels, init_theta, _data_val, labels_val, training_perf.col(ii));
            thetas.row(ii) = new_theta.t();
//            hconcat(new_theta.t(), thetas.row(ii));
            ii += 1;
        }

//        train_multiclass( data_t, labels_l, _data_val, _labels_val,
//                thetas, init_theta, params.train_parallel );
    }

    // check that the estimates are stable and finite
    this->learnt_thetas = thetas.clone();
    if( cvIsNaN( (double)sum(this->learnt_thetas)[0] ) )
    {
        CV_Error( CV_StsBadArg, "check training parameters. Invalid training classifier" );
    }

    // success
    ok = true;
    return ok;
}

//void LogisticRegressionImpl::train_multiclass( const Mat & data_t,
//        const Mat & labels_l, const Mat & _data_val, const Mat & _labels_val,
//        const Mat & thetas, const Mat & init_theta, bool parallelize ) {
//    Mat new_theta;
//    Mat labels_binary, labels_bin_val;
//    Mat labels, labels_val;
//    int ii = 0;
//    for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
//    {
//        // one-vs-rest (OvR) scheme
//        labels_binary = (labels_l == it->second)/255;
//        labels_binary.convertTo(labels, CV_32F);
//
//        labels_bin_val = (_labels_val == it->second)/255;
//        labels_bin_val.convertTo(labels_val, CV_32S);
//
//        if( params.train_method == BATCH)
//            new_theta = batch_gradient_descent(data_t, labels, init_theta, _data_val, labels_val, training_perf.col(ii));
//        else
//            new_theta = mini_batch_gradient_descent(data_t, labels, init_theta, _data_val, labels_val, training_perf.col(ii));
//        thetas.row(ii) = new_theta.t();
////            hconcat(new_theta.t(), thetas.row(ii));
//        ii += 1;
//    }
//}

float LogisticRegressionImpl::predict(InputArray samples, OutputArray results, int flags) const
{
    // check if learnt_mats array is populated
    if(!this->isTrained())
    {
        CV_Error( CV_StsBadArg, "classifier should be trained first" );
    }

    return compute_prediction( samples, learnt_thetas, results, flags );
}


float LogisticRegressionImpl::compute_prediction(InputArray samples,
        const Mat& _thetas, OutputArray results, int flags ) const
{
    // coefficient matrix
    Mat thetas;
    if ( _thetas.type() == CV_32F )
    {
        thetas = _thetas;
    }
    else
    {
        _thetas.convertTo( thetas, CV_32F );
    }
    CV_Assert(thetas.rows > 0);

    // data samples
    Mat data = samples.getMat();
    if(data.type() != CV_32F)
    {
        CV_Error( CV_StsBadArg, "data must be of floating type" );
    }

    // add a column of ones to the data (bias/intercept term)
    Mat data_t;
    hconcat( cv::Mat::ones( data.rows, 1, CV_32F ), data, data_t );
//    Mat data_t = data;
    CV_Assert(data_t.cols == thetas.cols);
//    CV_Assert(data_t.cols+1 == thetas.cols);

    // predict class labels for samples (handles binary and multiclass cases)
    Mat labels_c(data_t.rows, 1, CV_32S);
    Mat pred_m;
    Mat temp_pred;
    if(thetas.rows == 1)
    {
        // apply sigmoid function
//        temp_pred = calc_sigmoid(data_t * thetas.t());
        temp_pred = calc_sigmoid(data * thetas.colRange(1, thetas.cols).t() +
                thetas.at< float >( 0, 0 ) );

        CV_Assert(temp_pred.cols==1);
        pred_m = temp_pred.clone();

        // if greater than 0.5, predict class 0 or predict class 1
        temp_pred = (temp_pred > 0.5f) / 255;
        temp_pred.convertTo(labels_c, CV_32S);
    }
    else
    {
        // apply sigmoid function
        pred_m.create(data_t.rows, thetas.rows, data.type());
        for(int i = 0; i < thetas.rows; i++)
        {
//            temp_pred = calc_sigmoid(data_t * thetas.row(i).t());
            temp_pred = calc_sigmoid(data * thetas.row(i).colRange(1, thetas.cols).t() +
                    thetas.row(i).at< float >( 0, 0 ) );

            vconcat(temp_pred, pred_m.col(i));
        }

        // predict class with the maximum output
        Point max_loc;
        for(int i = 0; i < pred_m.rows; i++)
        {
            temp_pred = pred_m.row(i);
            minMaxLoc( temp_pred, NULL, NULL, NULL, &max_loc );
            labels_c.at< int >(i, 0) = max_loc.x;
//            labels.push_back(max_loc.x);
        }
    }

    // return label of the predicted class. class names can be 1,2,3,...
    remap_labels(labels_c, this->reverse_mapper);
//    pred_labs.convertTo(pred_labs, CV_32S);

    // return either the labels or the raw output
    if ( results.needed() )
    {
        if ( flags & StatModel::RAW_OUTPUT )
        {
            pred_m.copyTo( results );
        }
        else
        {
            labels_c.copyTo(results);
        }
    }

    return ( labels_c.empty() ? 0.f : static_cast<float>(labels_c.at<int>(0)) );
}


Mat LogisticRegressionImpl::calc_sigmoid(const Mat& data) const
{
    Mat dest;
    exp(-data, dest);
    return 1.0/(1.0+dest);
}

double LogisticRegressionImpl::compute_cost(const Mat& _data, const Mat& _labels, const Mat& _init_theta)
{
    int llambda = 0;
    int m;
    int n;
    double cost = 0;
    double rparameter = 0;
    Mat theta_b;
    Mat theta_c;
    Mat d_a;
    Mat d_b;

    m = _data.rows;
    n = _data.cols;

    theta_b = _init_theta(Range(1, n), Range::all());

    if (params.norm != REG_DISABLE)
    {
        llambda = 1;
    }

    if(this->params.norm == LogisticRegression::REG_L1)
    {
        rparameter = (llambda/(2*m)) * sum(theta_b)[0];
    }
    else
    {
        // assuming it to be L2 by default
        multiply(theta_b, theta_b, theta_c, 1);
        rparameter = (llambda/(2*m)) * sum(theta_c)[0];
    }

    d_a = calc_sigmoid(_data * _init_theta);
    log(d_a, d_a);
    multiply(d_a, _labels, d_a);

    // use the fact that: log(1 - sigmoid(x)) = log(sigmoid(-x))
    d_b = calc_sigmoid(- _data * _init_theta);
    log(d_b, d_b);
    multiply(d_b, 1-_labels, d_b);

    cost = (-1.0/m) * (sum(d_a)[0] + sum(d_b)[0]);
    cost = cost + rparameter;

    if(cvIsNaN( cost ) == 1)
    {
        CV_Error( CV_StsBadArg, "check training parameters. Invalid training classifier" );
    }

    return cost;
}


void LogisticRegressionImpl::compute_gradient(const Mat& _data, const Mat& _labels, const Mat &_theta, const double _lambda, Mat & _gradient )
{
    const int m = _data.rows;
    Mat pcal_a, pcal_b, pcal_ab;

    const Mat z = _data * _theta;

    CV_Assert( _gradient.rows == _theta.rows && _gradient.cols == _theta.cols );

    pcal_a = calc_sigmoid(z) - _labels;
    pcal_b = _data(Range::all(), Range(0,1));
    multiply(pcal_a, pcal_b, pcal_ab, 1);

    _gradient.row(0) = ((float)1/m) * sum(pcal_ab)[0];

    //cout<<"for each training data entry"<<endl;
    for(int ii = 1;ii<_gradient.rows;ii++)
    {
        pcal_b = _data(Range::all(), Range(ii,ii+1));
        multiply(pcal_a, pcal_b, pcal_ab, 1);

        _gradient.row(ii) = (1.0/m)*sum(pcal_ab)[0] + (_lambda/m) * _theta.row(ii);
    }
}


Mat LogisticRegressionImpl::batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta, const Mat& _data_val, const Mat& _labels_val, Mat2f perf)
{
    // implements batch gradient descent
    if(this->params.alpha<=0)
    {
        CV_Error( CV_StsBadArg, "check training parameters (learning rate) for the classifier" );
    }

    if(this->params.num_iters <= 0)
    {
        CV_Error( CV_StsBadArg, "number of iterations cannot be zero or a negative number" );
    }

    CV_Assert(perf.rows >= params.num_iters && perf.cols >= 1);

    int llambda = 0;
    int m;
    Mat theta_p = _init_theta.clone();
    Mat gradient( theta_p.rows, theta_p.cols, theta_p.type() );
    m = _data.rows;

    Mat pred_res, theta_best;
    float acc = 0.0, best_acc = -1.0;
    double cost = 0.0;
    int no_improve_count = 0;

    if (params.norm != REG_DISABLE)
    {
        llambda = 1;
    }

    int last_recorded = 0;
    for(int i = 0; ; i++)
    {
        cost = compute_cost(_data, _labels, theta_p);

        // compute accuracy on the validation set and halt training if converged
        if ( !_data_val.empty() ) {
            compute_prediction( _data_val, theta_p, pred_res, 0 );
            acc = 1.0f - ( countNonZero( pred_res - _labels_val ) / _labels_val.total() );
            if ( acc > ( best_acc + 1e-2 ) ) {
                no_improve_count = 0;
                best_acc = acc;
                theta_p.copyTo( theta_best );
            } else if ( ++no_improve_count > this->params.max_iters_no_improvement ) {
                break;
            }
        }

        if ( i % params.record_train_freq == 0 ) {
            last_recorded = i / params.record_train_freq;
            perf( last_recorded, 0 )[0] = cost;
            perf( last_recorded, 0 )[1] = acc;
        }

        if ( i >= this->params.num_iters ) break;

        compute_gradient( _data, _labels, theta_p, llambda, gradient );

        theta_p = theta_p - ( static_cast<double>(this->params.alpha)/m)*gradient;
    }

    perf( last_recorded+1, 0 )[0] = cost;
    perf( last_recorded+1, 0 )[1] = acc;

    return theta_best;
}

Mat LogisticRegressionImpl::mini_batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta, const Mat& _data_val, const Mat& _labels_val, Mat2f perf)
{
    // implements batch gradient descent
    int lambda_l = 0;
    int m;
    int j = 0;
    int size_b = this->params.mini_batch_size;

    if(this->params.mini_batch_size <= 0 || this->params.alpha == 0)
    {
        CV_Error( CV_StsBadArg, "check training parameters for the classifier" );
    }

    if(this->params.num_iters <= 0)
    {
        CV_Error( CV_StsBadArg, "number of iterations cannot be zero or a negative number" );
    }

    CV_Assert( perf.cols == 1 );

    Mat theta_p = _init_theta.clone();
    Mat gradient( theta_p.rows, theta_p.cols, theta_p.type() );
    Mat data_d;
    Mat labels_l;

    Mat pred_res, theta_best;
    float acc = 0.0, best_acc = 0.0;
    double cost = 0.0;
    int no_improve_count = 0;
    int max_NI_iters = this->params.max_iters_no_improvement;

    if (params.norm != REG_DISABLE)
    {
        lambda_l = 1;
    }

    // compute the initial validation set accuracy
    if ( !_data_val.empty() ) {
        compute_prediction( _data_val, theta_p.t(), pred_res, 0 );
        acc = 1.0f - ( (float) countNonZero( pred_res - _labels_val ) ) /
                ( (float) _labels_val.total() );
        best_acc = acc;
    }

    double alpha = static_cast<double>(this->params.alpha);

    int last_recorded = 0;
    for(int i = 0; ;i++)
    {
        if(j + size_b <= _data.rows)
        {
            data_d = _data(Range(j,j+size_b), Range::all());
            labels_l = _labels(Range(j,j+size_b),Range::all());
        }
        else
        {
            data_d = _data(Range(j, _data.rows), Range::all());
            labels_l = _labels(Range(j, _labels.rows),Range::all());
        }

        m = data_d.rows;

        if ( ( i % params.record_train_freq ) == 0 ) {
            cost = compute_cost(data_d, labels_l, theta_p);
            last_recorded = i / params.record_train_freq;
            perf( last_recorded, 0 )[0] = cost;
            perf( last_recorded, 0 )[1] = acc;
        }

        if ( i >= this->params.term_crit.maxCount ) break;

        compute_gradient(data_d, labels_l, theta_p, lambda_l, gradient);

        theta_p = theta_p - (alpha / m)*gradient;

        j += this->params.mini_batch_size;

        // if iterated through all training data rows
        if (j >= _data.rows) {
            j = 0;

            // compute accuracy on the validation set and halt training if converged
            if ( !_data_val.empty() ) {
                compute_prediction( _data_val, theta_p.t(), pred_res, 0 );
                acc = 1.0f - ( (float) countNonZero( pred_res - _labels_val ) ) /
                        ( (float) _labels_val.total() );
                if ( acc >= ( best_acc + 1e-2 ) ) {
                    best_acc = acc;
                    no_improve_count = 0;
                    theta_p.copyTo( theta_best );
                } else {
                    ++no_improve_count;
                    if ( params.decrease_alpha && no_improve_count == max_NI_iters / 2 ) {
                        alpha /= 2.0;
                        std::cout << "LR (mini batch) reducing learning rate to " << alpha << " after " << i << " iterations" << endl;
                    }
                    if ( no_improve_count > max_NI_iters ) {
                        std::cout << "logistic regression (mini batch) stopping early after " << i << " iterations" << endl;
                        break;
                    }
                }
            }
        }
    }

    cost = compute_cost(data_d, labels_l, theta_p);
    compute_prediction( _data_val, theta_p.t(), pred_res, 0 );
    acc = 1.0f - ( (float) countNonZero( pred_res - _labels_val ) ) /
            ( (float) _labels_val.total() );

    perf( last_recorded+1, 0 )[0] = cost;
    perf( last_recorded+1, 0 )[1] = acc;

    return theta_best;
}

bool LogisticRegressionImpl::set_label_map(const Mat &_labels_i)
{
    // this function creates two maps to map user defined labels to program friendly labels two ways.
    int ii = 0;
    Mat labels;

    this->labels_o = Mat(0,1, CV_8U);
    this->labels_n = Mat(0,1, CV_8U);

    _labels_i.convertTo(labels, CV_32S);

    for(int i = 0;i<labels.rows;i++)
    {
        this->forward_mapper[labels.at<int>(i)] += 1;
    }

    for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
    {
        this->forward_mapper[it->first] = ii;
        this->labels_o.push_back(it->first);
        this->labels_n.push_back(ii);
        ii += 1;
    }

    for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
    {
        this->reverse_mapper[it->second] = it->first;
    }

    return true;
}

void LogisticRegressionImpl::remap_labels(Mat & labels_i, const map<int, int>& lmap) const
{
    CV_Assert( labels_i.type() == CV_32S );
    CV_Assert( !lmap.empty() );

    for(int i =0;i<labels_i.rows;i++)
    {
        labels_i.at<int>(i,0) = lmap.find(labels_i.at<int>(i,0))->second;
    }
}

void LogisticRegressionImpl::clear()
{
    this->learnt_thetas.release();
    this->labels_o.release();
    this->labels_n.release();
}

void LogisticRegressionImpl::write(FileStorage& fs) const
{
    // check if open
    if(fs.isOpened() == 0)
    {
        CV_Error(CV_StsBadArg,"file can't open. Check file path");
    }
    string desc = "Logistic Regression Classifier";
    fs<<"classifier"<<desc.c_str();
    fs<<"alpha"<<this->params.alpha;
    fs<<"iterations"<<this->params.num_iters;
    fs<<"norm"<<this->params.norm;
    fs<<"train_method"<<this->params.train_method;
    if(this->params.train_method == LogisticRegression::MINI_BATCH)
    {
        fs<<"mini_batch_size"<<this->params.mini_batch_size;
    }
    fs<<"use_validation"<<this->params.test_data_is_validation;
    fs<<"max_no_improve_iters"<<this->params.max_iters_no_improvement;
    fs<<"learnt_thetas"<<this->learnt_thetas;
    fs<<"n_labels"<<this->labels_n;
    fs<<"o_labels"<<this->labels_o;
}

void LogisticRegressionImpl::read(const FileNode& fn)
{
    // check if empty
    if(fn.empty())
    {
        CV_Error( CV_StsBadArg, "empty FileNode object" );
    }

    this->params.alpha = (double)fn["alpha"];
    this->params.num_iters = (int)fn["iterations"];
    this->params.norm = (int)fn["norm"];
    this->params.train_method = (int)fn["train_method"];

    if(this->params.train_method == LogisticRegression::MINI_BATCH)
    {
        this->params.mini_batch_size = (int)fn["mini_batch_size"];
    }

    if ( !fn["use_validation"].empty() )
        fn["use_validation"] >> this->params.test_data_is_validation;

    if ( !fn["max_no_improve_iters"].empty() )
        fn["max_no_improve_iters"] >> this->params.max_iters_no_improvement;

    fn["learnt_thetas"] >> this->learnt_thetas;
    fn["o_labels"] >> this->labels_o;
    fn["n_labels"] >> this->labels_n;

    for(int ii =0;ii<labels_o.rows;ii++)
    {
        this->forward_mapper[labels_o.at<int>(ii,0)] = labels_n.at<int>(ii,0);
        this->reverse_mapper[labels_n.at<int>(ii,0)] = labels_o.at<int>(ii,0);
    }
}

}
}

/* End of file. */
