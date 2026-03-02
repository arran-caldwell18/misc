#include <stdlib.h>
#include <math.h>
#include <time.h>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double derivative_sigmoid(double x)
{
    return x * (1 - x);
}

//generate all random w , b
void init_random(double *w, double *b) {
    for (int i = 0; i < 6; i++) w[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

    for (int i = 0; i < 3; i++) b[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

//set random seed
void seed() {
    srand(time(NULL));
}

void train_model(double *w , double *b , int epochs , double learning_rate)
{
  //training date
  double training_data[4][3] = {
        {0,0,0},
        {0,1,1},
        {1,0,1},
        {1,1,0}
    };

    //dynamic allocation of shape of training data
    int cols = sizeof(training_data) / sizeof(training_data[0]);

    // loops for however many epochs is given when calling in python
    for (int e = 0; e < epochs; e++)
    {
        for (int i = 0; i < cols; i++)
        {
            //looping thorught training data
            double x1 = training_data[i][0];
            double x2 = training_data[i][1];
            double y  = training_data[i][2];

            //    FORWARD PASS
            double z1 = w[0]*x1 + w[1]*x2 + b[0];
            double a1 = sigmoid(z1);

            double z2 = w[2]*x1 + w[3]*x2 + b[1];
            double a2 = sigmoid(z2);

            double z3 = w[4]*a1 + w[5]*a2 + b[2];
            double y_hat = sigmoid(z3);

            //    BACKPROP
            double error = y_hat -y;
            double d_yhat = error;
            double d_z3 = d_yhat * derivative_sigmoid(y_hat);

            //output layer gradients
            double dw5 = d_z3 * a1;
            double dw6 = d_z3 * a2;
            double db3 = d_z3;

            double d_a1 = d_z3 * w[4];
            double d_a2 = d_z3 * w[5];

            double d_z1 = d_a1 * derivative_sigmoid(a1);
            double d_z2 = d_a2 * derivative_sigmoid(a2);

            double dw1 = d_z1 * x1;
            double dw2 = d_z1 * x2;
            double db1 = d_z1;
            double dw3 = d_z2 * x1;
            double dw4 = d_z2 * x2;
            double db2 = d_z2;

            w[0] -= learning_rate * dw1;
            w[1] -= learning_rate * dw2;
            w[2] -= learning_rate * dw3;
            
            w[3] -= learning_rate * dw4;
            w[4] -= learning_rate * dw5;
            w[5] -= learning_rate * dw6;

            b[0] -= learning_rate * db1;
            b[1] -= learning_rate * db2;
            b[2] -= learning_rate * db3;
        }
    }
}
