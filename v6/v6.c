//Importvements:
//dynamic input
//names file name at the top which makes it easier
// v7 big imporvs coming v6 is mostly js a fix up
//got GPt to fix all variable names since i cba
 
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
 
//max sizes of 2d array
//slightly bigger cols and rows since more data
#define MAX_ROWS 1000
#define MAX_COLS 20
#define MAX_LEN 100
 
char *file_name = "Training_data.csv";
 
typedef struct
{
    //multiple w per neuron thats why pointer
    double *weights;
    double bias;
    double output;
    //delta is used as a error signal for a neuron
    double delta;
} Neuron;
 
typedef struct
{
    //a layer has multiple neurons
    Neuron *neurons;
    int size;
} Layer;
 
typedef struct
{
    //each network is made up of layers
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
 
 
void init_network(Network *net, int *layer_sizes, int num_layers)
{
    //setting the variables we already know to the network struct variables
    net->num_layers = num_layers;
    //making the layer variable in the Network struct the size of how many layers we have by the size of a Layer
    net->layers = malloc(num_layers * sizeof(Layer));
 
    //for every layer
    for (int i = 0; i < num_layers; i++)
    {
        //setting the networks layer size to the correct size
        net->layers[i].size = layer_sizes[i];
        net->layers[i].neurons = malloc(layer_sizes[i] * sizeof(Neuron));
 
        //for every neuron
        for (int j = 0; j < layer_sizes[i]; j++)
        {
            //input layer has no weights
            //dont know what this line does i think it finds if this is the input neuron and if it is it doesnt have any biases or something
            int num_inputs = (i == 0) ? 0 : layer_sizes[i - 1];
 
            //allocating each weight the size of a double in memory
            net->layers[i].neurons[j].weights = malloc(num_inputs * sizeof(double));
            //sets random bias for the network to adjust later in training
            net->layers[i].neurons[j].bias = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
 
            //setting the output and delta to zero (will be adjusted later just to get rid of garbage values)
            net->layers[i].neurons[j].output = 0.0;
            net->layers[i].neurons[j].delta = 0.0;
 
            //setting weights needs to be in a loop since it depends on how many input neurons we have
            for (int k = 0; k < num_inputs; k++)
            {
                net->layers[i].neurons[j].weights[k] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            }
        }
    }
}
 
void forward_pass(Network *net, double *inputs)
{
    for (int i = 0; i < net->layers[0].size; i++)
    {
        //sets input layer outputs directly from inputs
        net->layers[0].neurons[i].output = inputs[i];
    }
 
    //start from layer 1 skipping input layer
    for (int i = 1; i < net->num_layers; i++)
    {
        for (int j = 0; j < net->layers[i].size; j++)
        {
            double sum = net->layers[i].neurons[j].bias;
 
            //sum weighted output from prev layer (aka z)
            for (int k = 0; k < net->layers[i - 1].size; k++)
            {
                sum += net->layers[i].neurons[j].weights[k] * net->layers[i - 1].neurons[k].output;
            }
 
            net->layers[i].neurons[j].output = sigmoid(sum);
        }
    }
}
 
void backprop(Network *net, double target, double lr)
{
    //calculating the output layer delta
    int last = net->num_layers - 1;
    for (int i = 0; i < net->layers[last].size; i++)
    {
        double output = net->layers[last].neurons[i].output;
        double error = output - target;
        net->layers[last].neurons[i].delta = error * derivative_sigmoid(output);
    }
 
    //calculating hidden layer deltas(backwards (why called back prop))
    for (int i = last - 1; i > 0; i--)
    {
        for (int j = 0; j < net->layers[i].size; j++)
        {
            double output = net->layers[i].neurons[j].output;
            double error = 0.0;
 
            //sum delta * w from next layer
            for (int k = 0; k < net->layers[i + 1].size; k++)
            {
                error += net->layers[i + 1].neurons[k].delta * net->layers[i + 1].neurons[k].weights[j];
            }
 
            //actually calculation to find delta
            net->layers[i].neurons[j].delta = error * derivative_sigmoid(output);
        }
    }
 
    //updating w and b's
    for (int i = 1; i < net->num_layers; i++)
    {
        for (int j = 0; j < net->layers[i].size; j++)
        {
            for (int k = 0; k < net->layers[i - 1].size; k++)
            {
                //updating weights
                net->layers[i].neurons[j].weights[k] -= lr * net->layers[i].neurons[j].delta * net->layers[i - 1].neurons[k].output;
            }
            //updating biases
            net->layers[i].neurons[j].bias -= lr * net->layers[i].neurons[j].delta;
        }
    }
}
 
void free_network(Network *net)
{
    //free all allocated memory, good practice
    for (int i = 0; i < net->num_layers; i++)
    {
        for (int j = 0; j < net->layers[i].size; j++)
        {
            //free weights inside each neuron first
            free(net->layers[i].neurons[j].weights);
        }
        //free the neurons inside each layer
        free(net->layers[i].neurons);
    }
    //freeing the layers array
    free(net->layers);
}
 
 
int main()
{
    //open csv first so we can pre-scan the header before building the network
    FILE *file = fopen(file_name, "r");
    if (file == NULL)
    {
        //exit function early if problem
        printf("Error opening file\n");
        return 1;
    }
 
    //pre-scan header row to count columns
    //start at 1 because n commas = n+1 columns
    int total_cols = 1;
    char header[1024];
    fgets(header, sizeof(header), file); //reads just the first line, advances file pointer past it so the loading loop skips it
 
    for (int i = 0; header[i] != '\0'; i++) //keeps going until null
    {
        if (header[i] == ',') total_cols++;
    }
 
    //getting user input for which column the target is
    int target_col = 0;
    printf("Enter target column number (1-indexed): ");
    if (scanf("%d", &target_col) != 1)
    {
        printf("Invalid input!\n");
        return 1;
    }
    //make to 0-based index
    target_col -= 1;
 
    //admin with randomizer
    srand(time(NULL));
 
    //network shape (dynamic)
    int num_inputs = total_cols - 1;
    int layer_sizes[] = {num_inputs, 4, 1}; //num_inputs input, 4 hidden, 1 output
    int num_layers = sizeof(layer_sizes) / sizeof(int);
 
    Network net;
    //initializing network
    init_network(&net, layer_sizes, num_layers);
 
    //load csv (took from v4)
    char training_data[MAX_ROWS][MAX_COLS][MAX_LEN];
    char line[1024];
    int num_rows = 0;
 
    //looping over everything in the csv
    while (fgets(line, sizeof(line), file) && num_rows < MAX_ROWS)
    {
        //removing \n
        line[strcspn(line, "\n")] = 0;
 
        //setting to zero for each iteration
        int col = 0;
 
        //checking for when there is a comma (to separate values)
        char *token = strtok(line, ",");
 
        while (token != NULL && col < MAX_COLS)
        {
            strncpy(training_data[num_rows][col], token, MAX_LEN - 1);
            training_data[num_rows][col][MAX_LEN - 1] = '\0';
            token = strtok(NULL, ",");
            col++;
        }
        num_rows++;
    }
    fclose(file);
 
    //train
    int epochs = 50000;
    double lr = 0.1;
 
    //so basically looping over the network
    for (int e = 0; e < epochs; e++)
    {
        for (int i = 0; i < num_rows; i++)
        {
            //dynamic inputs, skip target column
            double inputs[num_inputs];
            int input_idx = 0;
            for (int col = 0; col < total_cols; col++)
            {
                if (col != target_col) inputs[input_idx++] = atof(training_data[i][col]);
            }
 
            forward_pass(&net, inputs);
            backprop(&net, atof(training_data[i][target_col]), lr);
        }
    }
 
    int correct = 0;
    double mse = 0.0;
 
    //evaluate
    for (int i = 0; i < num_rows; i++)
    {
        //dynamic inputs, skip target column
        double inputs[num_inputs];
        int input_idx = 0;
        for (int col = 0; col < total_cols; col++)
        {
            if (col != target_col) inputs[input_idx++] = atof(training_data[i][col]);
        }
 
        forward_pass(&net, inputs);
        double output = net.layers[net.num_layers - 1].neurons[0].output;
 
        //if (output >= 0.5) pred = 1; else pred = 0;
        int pred = output >= 0.5 ? 1 : 0;
 
        //calc accuracy
        double true_val = atof(training_data[i][target_col]);
        int true_label = (int)round(true_val);
        correct += (pred == true_label);
        mse += (output - true_val) * (output - true_val);
 
        //print all inputs dynamically
        printf("Inputs: (");
        for (int j = 0; j < num_inputs; j++)
        {
            printf("%.3f", inputs[j]);
            if (j < num_inputs - 1) printf(", ");
        }
        printf(") --- Pred: %d --- Raw: %.5f\n", pred, output);
    }
 
    //print model accuracy
    mse /= num_rows;
    printf("\nAccuracy: %.1f%% --- MSE: %.4f\n", (double)correct / num_rows * 100, mse);
 
    //free network
    free_network(&net);
    return 0;
}