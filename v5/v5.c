#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

//max sizes of 2d array
#define Max_Row 100
#define Max_Cols 10
#define Max_Len 100

typedef struct
{
    //mutltiple w per neruon thats why ointer
    double *wieghts;
    double bias;
    double output;
    //delta is used as a error signal for a nueron
    double delta;
} Neuron;

typedef struct 
{
    //a layer has nutiple nuerons
    Neuron *nuerons;
    int size;
} Layer;

typedef struct
{
    //each netowrk is made up of layers
    Layer *layers;
    int num_layers;
} Network;




double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double derivative_sigmoid(double x)
{
    return x * (1 - x);
}



void init_network(Network *net , int *layer_sizes , int num_layers)
{
    //setting the variables we already know to the network struct variables
    net -> num_layers = num_layers;
        //making teh layer varable in the Network sruct the size of how many layers we have by the size of a Layer
    net -> layers = malloc(num_layers * sizeof(Layer));


//for every layer
    for (int i = 0;i < num_layers; i++)
    {
        //setting the netowkrs layers size to the correct size
        net -> layers[i].size = layer_sizes[i];
        net -> layers[i].nuerons = malloc(layer_sizes[i] * sizeof(Neuron));


//for every nueron
        for (int j = 0; j < layer_sizes[i]; j++)
        {
            //input layer has no wieghts
            //dont know what this line does i think it finds if this is teh input nueron and if it is it doesnt have any biases or somthibg
            int num_inputs = (i == 0) ? 0 : layer_sizes[i - 1];

            //allocating each wieght the size of a double in memory
            net -> layers[i].nuerons[j].wieghts = malloc(num_inputs * sizeof(double));
            //sets random bias for the netowrk to ajust later in training
            net -> layers[i].nuerons[j].bias = ((double)rand()/RAND_MAX) * 2.0 - 1.0;

            //setting the output and delta to zero (will be ajusted later just to get rid of garbage values)
            net -> layers[i].nuerons[j].output = 0.0;
            net -> layers[i].nuerons[j].delta = 0.0;

            //setting biases needs to be in a loop since it depends on how many input nuerons we have
            for (int k = 0; k < num_inputs; k++)
            {
                net-> layers[i].nuerons[j].wieghts[k] = ((double)rand()/RAND_MAX) * 2.0 - 1.0;
            }
        }
    }
}

void forward_pass(Network *net , double *inputs)
{
    for (int i = 0; i < net -> layers[0].size; i++)
    {
        //sets input layer outputs directoly from inputs
        net -> layers[0].nuerons[i].output = inputs[i];
    }

    //start from layer 1 skipping input layer
    for (int i = 1; i < net -> num_layers; i++)
    {
        for (int j = 0; j < net->layers[i].size; j++)
        {
            double sum = net -> layers[i].nuerons[j].bias;

            //sum wieghted output from prev layer (aka z)
            for (int k = 0; k < net -> layers[i-1].size; k++)
            {
                sum += net->layers[i].nuerons[j].wieghts[k] * net -> layers[i-1].nuerons[k].output;
            }

            net -> layers[i].nuerons[j].output = sigmoid(sum);
        }
    }
}

void backprop(Network *net , double target , double lr)
{
    //calculating the output layer delta
    int last = net->num_layers - 1;
    for (int i = 0; i < net -> layers[last].size; i++)
    {
        double output = net -> layers[last].nuerons[i].output;
        double error = output - target;
        net -> layers[last].nuerons[i].delta = error * derivative_sigmoid(output);
    }

    //calculating hidden layer deltas(backwards (why called back prop))
    for (int i = last-1; i > 0; i--)
    {
        for (int j = 0; j < net->layers[i].size; j++)
        {
            double output = net->layers[i].nuerons[j].output;
            double error = 0.0;

            //sum delta * w from next layer
            for (int k = 0; k < net->layers[i+1].size; k++)
            {
                error += net->layers[i+1].nuerons[k].delta * net->layers[i+1].nuerons[k].wieghts[j];
            }

            //actually calculation to find delta
            net->layers[i].nuerons[j].delta = error * derivative_sigmoid(output);
        }
    }

    //updatdating w and b's
    for (int i = 1; i < net -> num_layers; i++)
    {
        for (int j = 0; j < net->layers[i].size; j++)
        {
            for (int k = 0; k < net->layers[i-1].size; k++)
            {
                //updating weigts
                net->layers[i].nuerons[j].wieghts[k] -= lr * net->layers[i].nuerons[j].delta * net->layers[i-1].nuerons[k].output;
            }
            //updating biases
            net -> layers[i].nuerons[j].bias -= lr * net -> layers[i].nuerons[j].delta;
        }
    }
}

void free_network(Network *net)
{
    //free all allocated memory sa good practise
    for (int i = 0; i< net->num_layers; i++)
    {
        for (int j = 0; j < net->layers[i].size;j++)
        {
            //free weights inside each neuron first
            free(net->layers[i].nuerons[j].wieghts);
        }
        //free the neurons inside each layer
        free(net->layers[i].nuerons);
    }
    //freeing the layers array
    free(net->layers);
} 




int main()
{
    //admin with randomizer
    srand(time(NULL));

    //netowrk shape
    int layer_size[] = {2,4,1};//2 input 4 hidden 1 output
    int num_layers = sizeof(layer_size) / sizeof(int);

    Network circle_net;
    //initializing netrok
    init_network(&circle_net , layer_size , num_layers);

    //load csv (took from v4)
    FILE *file = fopen("//home//arranthegreat//Desktop/PYTHON//neural network from scratch//v5//training_data.csv","r");
    if (file== NULL)
    {
        //exist fucntion early if problem
        printf("Error opening file: %s\n", "//home//arranthegreat//Desktop/PYTHON//neural network from scratch//v5//training_data.csv");
        return 1;
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




    //train
    int epochs = 50000;
    double lr = 0.1;

    //so basiaclly looping over the netowrk
    for (int e = 0 ; e < epochs ; e++)
    {
        for (int i =0;i<row ; i++)
        {
            double inputs[2] = {atof(training_data[i][0]), atof(training_data[i][1])};
            forward_pass(&circle_net , inputs);
            backprop(&circle_net, atof(training_data[i][2]), lr);
        }
    }

    int correct = 0;
    double mse = 0.0;

    //evaulate
    for (int i = 0; i < row; i++)
    {
        double inputs[2] = {atof(training_data[i][0]), atof(training_data[i][1])};
        forward_pass(&circle_net, inputs);
        double output = circle_net.layers[circle_net.num_layers-1].nuerons[0].output;

        //if (output >= 0.5) pred = 1; else pred = 0;
        int pred = output >= 0.5 ? 1 : 0;

        //calc accuracy
        int true_label = (int)round(atof(training_data[i][2]));
        correct += (pred == true_label);
        mse += (output - atof(training_data[i][2])) * (output - atof(training_data[i][2]));


        printf("Input: (%.3f, %.3f) --- Pred: %d --- Raw: %.5f\n",inputs[0], inputs[1], pred, output);
    }
    //rpint model accuracy
    mse /= row;printf("\nAccuracy: %.1f%% --- MSE: %.4f\n", (double)correct / row * 100, mse);

    //free network
    free_network(&circle_net);
    return 0;
}