#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

//max sizes of 2d array
#define Max_Row 100
#define Max_Cols 10
#define Max_Len 100

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

void train_model(double *w , double *b , int epochs , double learning_rate, char* file_path)
{
    //training data duyanmic from file location given in python
    FILE *file = fopen(file_path,"r");
    if (file== NULL)
    {
        //exist fucntion early if problem
        printf("Error opening file: %s\n", file_path);
        return;
    }

    //creating a 2d array from the csv
    char training_data[Max_Row][Max_Cols][Max_Len];
    char line[1024];
    int row = 0;

    //looping over everything in the csv
    while (fgets(line , sizeof(line),file) && row < Max_Row)
    {
        //removing \n
        line[strcspn(line, "\n")] = 0;

        //setting to zero for each iteration
        int col = 0;
        
        //checking for when there is a comma (to seperate values)
        char *token = strtok(line , ",");

        while (token != NULL && col < Max_Cols)
        {
            strncpy(training_data[row][col], token, Max_Len - 1);
            training_data[row][col][Max_Len - 1] = '\0';
            token = strtok(NULL,",");
            col++;
        }
        row++;
    }
    fclose(file);


    //dynamic allocation of shape of training data
    int rows = row;

    // loops for however many epochs is given when calling in python
    for (int e = 0; e < epochs; e++)
    {
        for (int i = 0; i < rows; i++)
        {
            //looping thorught training data
            double x1 = atof(training_data[i][0]);
            double x2 = atof(training_data[i][1]);
            double y  = atof(training_data[i][2]);

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
