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
        max_iters_no_improvement = 0;
        record_train_period = 300;
        train_parallel = false;
        decrease_alpha = false;
        use_one_vs_all = true;
    }

    double alpha; //!< learning rate.
    int num_iters; //!< number of iterations.
    int norm;
    int train_method;
    int mini_batch_size;
    TermCriteria term_crit;
    int max_iters_no_improvement; //!< max number of iterations to keep training if there's no improvement on validation set accuracy
    bool record_training; //!< true to record the cost and validation accuracy (if computed) for each training iteration
    int record_train_period; //!< record training performance at when (iter % record_train_period)==0
    bool train_parallel; //!< true to parallelize some computations during training (only beneficial with large training data)
    bool decrease_alpha; //!< true to decrease the learning rate if no progress is being made relative to the validation set
    bool use_one_vs_all; //!< true to use 1 vs all instead of multinomial logistic regression (i.e. softmax regression)
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
    CV_IMPL_PROPERTY(int, MaxItersNoValidImprovement, params.max_iters_no_improvement)
    CV_IMPL_PROPERTY(bool, RecordTrainingPerf, params.record_training)
    CV_IMPL_PROPERTY(int, TrainingPerfRecordPeriod, params.record_train_period)
    CV_IMPL_PROPERTY(bool, ParallelizeTraining, params.train_parallel)
    CV_IMPL_PROPERTY(bool, DecreaseAlpha, params.decrease_alpha)
    CV_IMPL_PROPERTY(bool, OneVsAll, params.use_one_vs_all)

    virtual bool train( const Ptr<TrainData>& trainData, int=0 );
    virtual float predict(InputArray samples, OutputArray results, int flags=0) const;
    virtual void clear();
    virtual void write(FileStorage& fs) const;
    virtual void read(const FileNode& fn);
    virtual Mat get_learnt_thetas() const { return learnt_thetas; }
    virtual Mat2f get_training_perf() const { return training_perf; }
    virtual void set_validation_data(const Ptr<const TrainData>& _validationData) { validationData = _validationData; }
    virtual int getVarCount() const { return learnt_thetas.cols; }
    virtual bool isTrained() const { return !learnt_thetas.empty(); }
    virtual bool isClassifier() const { return true; }
    virtual String getDefaultName() const { return "opencv_ml_lr"; }
protected:
    float compute_prediction(InputArray samples, const Mat& _thetas, OutputArray results, int flags ) const;
    float compute_prediction_accuracy(InputArray samples, const Mat& labels, const Mat& thetas, OutputArray results);
    Mat calc_sigmoid(const Mat& data) const;
    void compute_class_probabilities(const Mat& _data, const Mat& _theta, Mat& p_yc) const;
    double compute_cost(const Mat& _data, const Mat& _labels, const Mat& _theta, const Mat& _p_yc, const double _lambda) const;
    void compute_gradient(const Mat& _data, const Mat& _labels, const Mat &_theta, const Mat& _p_yc, const double _lambda, Mat & _gradient );
    void check_gradient(const Mat& _gradient, const Mat& _data, const Mat& _labels, const Mat& _theta, const double _lambda) const;
    Mat batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta, const Mat& _data_val, const Mat& _labels_val, Mat2f perf);
    Mat mini_batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta, const Mat& _data_val, const Mat& _labels_val, Mat2f perf);
    bool set_label_map(const Mat& _labels_i);
    void remap_labels(Mat& labels_i, const map<int, int>& lmap) const;
protected:
    LrParams params;
    Mat learnt_thetas;
    Mat2f training_perf;
    Ptr<const TrainData> validationData;
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
    if ( validationData && validationData->getNSamples() > 0 )
    {
        _data_val = validationData->getSamples();
        _labels_val = validationData->getResponses();
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

    const int m = _data_i.rows;
    const int n = _data_i.cols + 1;

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

    int num_perf_iters = max(params.num_iters, params.term_crit.maxCount) / params.record_train_period;

    // add a column of ones to the data (bias/intercept term)
    Mat data_t;
    hconcat( _data_i, cv::Mat::ones( _data_i.rows, 1, CV_32F ), data_t );

    // coefficient matrix (zero-initialized)
    Mat thetas;
    Mat init_theta = Mat::zeros(1, n, CV_32F);

    Mat new_theta;
    Mat labels, labels_val;

    // fit the model (handles binary and multiclass cases)
    if(num_classes == 2 || !params.use_one_vs_all)
    {
        thetas.create(num_classes-1, n, CV_32F);
        training_perf = Mat2f::zeros(2 + num_perf_iters, 1);
        init_theta = cv::repeat(init_theta, num_classes-1, 1);

        labels.create(m, num_classes, CV_32F);

        Mat ll;
        for(int ii = 0; ii < num_classes; ++ii) {
            ll = (labels_l == ii ) / 255;
            ll.convertTo(labels.col(ii), CV_32F);
        }

        _labels_val.convertTo(labels_val, CV_32S);

        if(this->params.train_method == LogisticRegression::BATCH)
            new_theta = batch_gradient_descent(data_t, labels, init_theta, _data_val, labels_val, training_perf);
        else
            new_theta = mini_batch_gradient_descent(data_t, labels, init_theta, _data_val, labels_val, training_perf);
        thetas = new_theta;
    }
    else
    {
        /* take each class and rename classes you will get a theta per class
        as in multi class class scenario, we will have n thetas for n classes */
        thetas.create(num_classes, n, CV_32F);

        training_perf = Mat2f::zeros(2 + num_perf_iters, num_classes);

        Mat labels_binary, labels_bin_val;
        int ii = 0;
        for(map<int,int>::iterator it = this->forward_mapper.begin(); it != this->forward_mapper.end(); ++it)
        {
            // one-vs-rest (OvR) scheme
            hconcat( (labels_l != it->second)/255, (labels_l == it->second)/255, labels_binary);
            labels_binary.convertTo(labels, CV_32F);

            if ( !_labels_val.empty() ) {
                labels_val = Mat::zeros(_labels_val.rows, 1, CV_32S);
                labels_val.setTo(reverse_mapper[1], _labels_val == it->first);
                labels_val.setTo(reverse_mapper[0], _labels_val != it->first);
            }

            if( params.train_method == BATCH)
                new_theta = batch_gradient_descent(data_t, labels, init_theta, _data_val, labels_val, training_perf.col(ii));
            else
                new_theta = mini_batch_gradient_descent(data_t, labels, init_theta, _data_val, labels_val, training_perf.col(ii));
            new_theta.copyTo(thetas.row(ii));
            ii += 1;
        }
    }

    // check that the estimates are stable and finite
    this->learnt_thetas = thetas.clone();
    if( cvIsNaN( (double)sum(this->learnt_thetas)[0] ) )
    {
        CV_Error( CV_StsBadArg, "check training parameters. Invalid learned classifier (thetas are NaN)" );
    }

    // success
    ok = true;
    return ok;
}


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

    // implicitly add a column of ones to the end of the data (bias/intercept term)
    CV_Assert(data.cols+1 == thetas.cols);

    // predict class labels for samples (handles binary and multiclass cases)
    Mat labels_c(data.rows, 1, CV_32S);
    Mat pred_m;
    Mat temp_pred;

    if(thetas.rows == 1 || !params.use_one_vs_all)
    {
        compute_class_probabilities(data, thetas, pred_m);
    }
    else
    {
        // apply sigmoid function
        pred_m.create(data.rows, thetas.rows, data.type());
        for(int i = 0; i < thetas.rows; i++)
        {
//            temp_pred = calc_sigmoid(data * thetas.row(i).colRange(0, lasttc).t() +
//                    thetas.at< float >( i, lasttc ) );
            compute_class_probabilities(data, thetas.row(i), temp_pred);
            temp_pred.col(1).copyTo(pred_m.col(i));
        }
    }

    // predict class with the maximum output
    Point max_loc;
    for(int i = 0; i < pred_m.rows; i++)
    {
        minMaxLoc( pred_m.row(i), NULL, NULL, NULL, &max_loc );
        labels_c.at< int >(i, 0) = max_loc.x;
    }

    // return label of the predicted class. class names can be 1,2,3,...
    remap_labels(labels_c, this->reverse_mapper);

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


float LogisticRegressionImpl::compute_prediction_accuracy(InputArray samples, const Mat& labels, const Mat& thetas, OutputArray results) {
    compute_prediction( samples, thetas, results, 0 );
    float acc = 1.0f - ( (float) countNonZero( results.getMatRef() - labels ) / labels.rows );
    CV_Assert( labels.cols == 1 );
    return acc;
}


Mat LogisticRegressionImpl::calc_sigmoid(const Mat& data) const
{
    Mat dest;
    exp(-data, dest);
    return 1.0/(1.0+dest);
}


struct LogisticRegressionImpl_ComputeClassProbs_Impl : ParallelLoopBody
{
    const Mat& data;
    const Mat& theta;
    Mat& p_yc;

    LogisticRegressionImpl_ComputeClassProbs_Impl(const Mat& _data, const Mat &_theta, Mat & _p_yc)
        : data(_data)
        , theta(_theta)
        , p_yc(_p_yc)
    { }

    void operator()(const cv::Range& r) const
    {
        const Mat data_r = data.rowRange(r.start, r.end);

        const int /*m = _data.rows,*/ K = theta.rows+1;
        const int ones_idx = theta.cols-1;
        Mat z, sum_z;//, maxvals;

        // implicitly add the intercept terms if they don't exist in the data
        if ( data_r.cols < theta.cols ) {
            z = data_r * theta.colRange(0, ones_idx).t();  // (MxN) x ((K-1)x((N+1)-1))^T = Mx(K-1)
            const Mat th_ones = theta.col( ones_idx ).t(); // ((K-1)x1)^T = 1x(K-1)
            for ( int i = 0; i < z.rows; ++i ) {
                z.row( i ) += th_ones;
            }
        } else {
            z = data_r * theta.t();  // (MxN+1) x ((K-1)x(N+1))^T = Mx(K-1)
        }

//    // subtract the largest element from each row to prevent overflow
//    reduce(z, maxvals, 1, REDUCE_MAX);
//    maxvals = max(maxvals, 0.0);
//    for (int i = 0; i < z.cols; ++i ) z.col(i) -= maxvals;

        // compute softmax
        exp(z, z);
        reduce(z, sum_z, 1, REDUCE_SUM);
        sum_z += 1.0;

//    exp(-maxvals, maxvals);
//    sum_z += maxvals;

        CV_Assert(z.cols == K-1);

        // normalize
        p_yc.rowRange(r.start, r.end).col(0) = 1.0 / sum_z; // add the column for the all-zero theta
        for (int i = 0; i < z.cols; ++i) {
            divide(z.col(i), sum_z, p_yc.rowRange(r.start, r.end).col(i+1));
        }
    }
};


void LogisticRegressionImpl::compute_class_probabilities( const Mat & _data,
        const Mat & _theta, Mat & p_yc ) const {

    CV_Assert( _data.cols+1 == _theta.cols || _data.cols == _theta.cols );

    const int m = _data.rows, K = _theta.rows+1;
    p_yc.create(m, K, CV_32F);

    LogisticRegressionImpl_ComputeClassProbs_Impl ccpi(_data, _theta, p_yc);

    if ( params.train_parallel ) {
        double nstripes = cv::getNumThreads();
        cv::parallel_for_(cv::Range(0, _data.rows), ccpi, nstripes);
    } else {
        ccpi(cv::Range(0, _data.rows));
    }
}


double LogisticRegressionImpl::compute_cost(const Mat& _data, const Mat& _labels, const Mat& _theta, const Mat& _p_yc, const double _lambda) const
{
    int m;
//    int n;
    double cost = 0;
    double rparameter = 0;
    Mat theta_b;
    Mat theta_c;

    m = _data.rows;
    theta_b = _theta;

    if(this->params.norm == LogisticRegression::REG_L1)
    {
        rparameter = (_lambda/(2*m)) * sum(theta_b)[0];
        CV_Error(CV_StsBadArg, "L1 regularization is not currently supported for logistic regression");
    }
    else
    {
        // assuming it to be L2 by default
        multiply(theta_b, theta_b, theta_c, 1);
        rparameter = (_lambda/(2*m)) * sum(theta_c)[0];
    }

    Mat log_pyc;
    log(_p_yc, log_pyc);
    multiply(_labels, log_pyc, log_pyc);
    cost = (-1.0 / m) * sum(log_pyc)[0] + rparameter;

    if(cvIsNaN( cost ) == 1)
    {
        CV_Error( CV_StsBadArg, "check training parameters. Invalid training classifier (cost is NaN)" );
    }

    return cost;
}


void LogisticRegressionImpl::compute_gradient(const Mat& _data, const Mat& _labels, const Mat &_theta, const Mat& _p_yc, const double _lambda, Mat & _gradient )
{
    const int m = _data.rows;

    CV_Assert( _gradient.rows == _theta.rows && _gradient.cols == _theta.cols );

    // we assume that the data has a column of ones included, and we ignore the
    // row of zeros in theta
    const int K = _p_yc.cols;
    _gradient = ( _p_yc.colRange(1, K) - _labels.colRange(1, K) ).t() * _data + _lambda*_theta;

//    check_gradient(_gradient, _data, _labels, _theta, _lambda);
}


void LogisticRegressionImpl::check_gradient(const Mat& _gradient, const Mat& _data, const Mat& _labels, const Mat& _theta, const double _lambda) const {
    const double h = 1e-3;
    CV_Assert(_theta.type() == CV_32F && _gradient.type() == CV_32F);
    Mat th = _theta.clone();
    Mat g(_gradient.rows, _gradient.cols, _gradient.type());
    Mat p;
    for ( MatIterator_<float> it = th.begin<float>(); it != th.end<float>(); ++it ){
        *it += h/2.0;
        compute_class_probabilities(_data, th, p );
        double cplus = compute_cost(_data, _labels, th, p, _lambda);
        *it -= h;
        compute_class_probabilities(_data, th, p );
        double cminus = compute_cost(_data, _labels, th, p, _lambda);
        *it += h/2.0;
        Point pp = it.pos();
        g.at<float>(pp.y, pp.x) = (float) ((cplus - cminus) / h);
    }

    Mat diff = _gradient - g;
    int maxidx[] = {-1,-1};
    double maxval = 10000;
    minMaxIdx(diff, NULL, &maxval, NULL, maxidx);
    if ( maxval >= 1e-3 ) {
//        cout << "diff: " << diff << endl;
        cout << "gradient difference detected: " << endl;
        cout << "  g(" << maxidx[0] << ", " << maxidx[1] << ") = " << _gradient.at<float>(maxidx[0], maxidx[1]) << endl;
        cout << "gfd(" << maxidx[0] << ", " << maxidx[1] << ") = " << g.at<float>(maxidx[0], maxidx[1]) << endl;
    }
//    CV_Assert(maxval < 1e-4);
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

    CV_Assert(perf.rows >= 1 + ( (float) params.num_iters) / params.record_train_period && perf.cols == 1);

    const double llambda = ( params.norm == REG_DISABLE ? 0 : 1 );
    int m;
    double alpha = this->params.alpha;
    Mat theta_p = _init_theta.clone();
    Mat gradient( theta_p.rows, theta_p.cols, theta_p.type() );
    Mat p_yc;
    m = _data.rows;

    Mat pred_res, theta_best;
    float acc = 0.0, best_acc = -1.0;
    double cost = 0.0;
    int no_improve_count = 0, last_recorded = -1;
    const int max_NI_iters = params.max_iters_no_improvement;

    for(int i = 0; i < this->params.num_iters; i++)
    {
        compute_class_probabilities(_data, theta_p, p_yc);
        cost = compute_cost(_data, _labels, theta_p, p_yc, llambda);

        // compute accuracy on the validation set and halt training if converged
        if ( !_data_val.empty() ) {
            acc = compute_prediction_accuracy(_data_val, _labels_val, theta_p, pred_res);
            if ( !params.use_one_vs_all ) {
                if ( acc >= best_acc ) {
                    best_acc = acc;
                    no_improve_count = 0;
                    theta_p.copyTo( theta_best );
                } else {
                    ++no_improve_count;
                    if ( params.decrease_alpha &&
                            (( no_improve_count % ( max_NI_iters / 4 )) == 0 )) {
                        alpha /= 10.0;
                    }
                    if ( max_NI_iters > 0 && no_improve_count > max_NI_iters ) {
                        break;
                    }
                }
            }
        }

        if ( i % params.record_train_period == 0 ) {
            last_recorded = i / params.record_train_period;
            perf( last_recorded, 0 )[0] = cost;
            perf( last_recorded, 0 )[1] = acc;
        }

        compute_gradient( _data, _labels, theta_p, p_yc, llambda, gradient );

        theta_p = theta_p - (alpha/m)*gradient;
    }

    perf( last_recorded+1, 0 )[0] = cost;
    perf( last_recorded+1, 0 )[1] = acc;

    return theta_best.empty() ? theta_p : theta_best;
}

Mat LogisticRegressionImpl::mini_batch_gradient_descent(const Mat& _data, const Mat& _labels, const Mat& _init_theta, const Mat& _data_val, const Mat& _labels_val, Mat2f perf)
{
    // implements batch gradient descent
    double lambda_l = 0;
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
    Mat p_yc;

    Mat pred_res, theta_best;
    float acc = 0.0f, best_acc = -1.0f;
    double cost = 0.0;
    int no_improve_count = 0, last_recorded = -1;
    const int max_NI_iters = this->params.max_iters_no_improvement;

    if (params.norm != REG_DISABLE)
    {
        lambda_l = 1;
    }

    double alpha = this->params.alpha;
    if ( !_data_val.empty() ) {
        acc = compute_prediction_accuracy( _data_val, _labels_val, theta_p, pred_res );
    }

    for(int i = 0; i < this->params.num_iters; i++)
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

        compute_class_probabilities(data_d, theta_p, p_yc);

        if ( ( i % params.record_train_period ) == 0 ) {
            cost = compute_cost(data_d, labels_l, theta_p, p_yc, lambda_l);

            // compute accuracy on the validation set and halt training if converged
            if ( !_data_val.empty() ) {
                acc = compute_prediction_accuracy(_data_val, _labels_val, theta_p, pred_res);
                if ( !params.use_one_vs_all ) {
                    if ( acc >= best_acc ) {
                        best_acc = acc;
                        no_improve_count = 0;
                        theta_p.copyTo( theta_best );
                    } else {
                        ++no_improve_count;
                        if ( params.decrease_alpha &&
                                (( no_improve_count % ( max_NI_iters/4 )) == 0 )) {
                            alpha /= 10.0;
                        }
                        if ( max_NI_iters > 0 &&
                                no_improve_count > max_NI_iters ) {
                            break;
                        }
                    }
                }
            }

            last_recorded = i / params.record_train_period;
            perf( last_recorded, 0 )[0] = cost;
            perf( last_recorded, 0 )[1] = acc;
        }

        compute_gradient(data_d, labels_l, theta_p, p_yc, lambda_l, gradient);
        theta_p = theta_p - (alpha / m)*gradient;

        j += this->params.mini_batch_size;

        // if iterated through all training data rows
        if (j >= _data.rows) {
            j = 0;
        }
    }

    compute_class_probabilities(data_d, theta_p, p_yc);
    cost = compute_cost(data_d, labels_l, theta_p, p_yc, lambda_l);
    if ( !_data_val.empty() ) {
        acc = compute_prediction_accuracy( _data_val, _labels_val, theta_p,
                pred_res );
    }

    perf( last_recorded+1, 0 )[0] = cost;
    perf( last_recorded+1, 0 )[1] = acc;

    return theta_best.empty() ? theta_p : theta_best;
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
        CV_Error(CV_StsBadArg,"file can't open. Check file path");

    string desc = "Logistic Regression Classifier";
    fs<<"classifier"<<desc.c_str();

    fs<<"alpha"<<this->params.alpha;
    fs<<"iterations"<<this->params.num_iters;
    fs<<"norm"<<this->params.norm;
    fs<<"train_method"<<this->params.train_method;
    if(this->params.train_method == LogisticRegression::MINI_BATCH)
        fs<<"mini_batch_size"<<this->params.mini_batch_size;
    fs<<"max_no_improve_iters"<<this->params.max_iters_no_improvement;
    fs<<"train_parallel"<<this->params.train_parallel;
    fs<<"decrease_learning_rate"<<this->params.decrease_alpha;
    fs<<"use_one_vs_all"<<this->params.use_one_vs_all;

    fs<<"learnt_thetas"<<this->learnt_thetas;
    fs<<"n_labels"<<this->labels_n;
    fs<<"o_labels"<<this->labels_o;
}

void LogisticRegressionImpl::read(const FileNode& fn)
{
    // check if empty
    if(fn.empty())
        CV_Error( CV_StsBadArg, "empty FileNode object" );

    this->params.alpha = (double)fn["alpha"];
    this->params.num_iters = (int)fn["iterations"];
    this->params.norm = (int)fn["norm"];
    this->params.train_method = (int)fn["train_method"];

    if (this->params.train_method == LogisticRegression::MINI_BATCH)
        this->params.mini_batch_size = (int)fn["mini_batch_size"];

    if (!fn["max_no_improve_iters"].empty())
        fn["max_no_improve_iters"] >> this->params.max_iters_no_improvement;

    if (!fn["train_parallel"].empty())
        fn["train_parallel"] >> this->params.train_parallel;

    if (!fn["decrease_learning_rate"].empty())
        fn["decrease_learning_rate"] >> this->params.decrease_alpha;

    if (!fn["use_one_vs_all"].empty())
        fn["use_one_vs_all"] >> this->params.use_one_vs_all;

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
