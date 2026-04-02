//IMPROVV
//
//MAIN -> Evoluationary (instead of hard coding the nn shape we use evolution to dynamically find the best shape for the dataset) 
//Hidden layer no longer hardcoded as a result

//note -im like 99% certain this works cuz ima uplaod it as the code is running but no errors and on gen 3

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
 
//max sizes of 2d array
//slightly bigger cols and rows since more data
#define MAX_ROWS 1050
#define MAX_COLS 20
#define MAX_LEN 200
 
//max amount of hidden layers we allow to stop infinite processing
#define MAX_HIDDEN 5
 
//max len of header to stop hazardous data
#define MAX_HEADER 1024
 
char *file_name = "training_data.csv";
 
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
 
//architecture
typedef struct
{
    int num_hidden_layers;          //how many hidden layers this particular architeture has
    int layer_sizes[MAX_HIDDEN];    //the nueron count for each layer
    double fitness;                 //where we store teh accuracy
} Architecture;
 
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

//freee net
void free_network(Network *net)
{
    for (int i = 0; i < net->num_layers; i++)
    {
        for (int j = 0; j < net->layers[i].size; j++)
        {
            free(net->layers[i].neurons[j].weights);
        }
        free(net->layers[i].neurons);
    }
    free(net->layers);
}
 
//building the layer sizes (helper)
 
//arch -> the like varibale 
//num_inputs -> number of input nuerons
//out -> the array we are building, caller passes this in empty and we fill it
//out_len -> we write the final legnth here so the caller knows gow long out is
//pointers since we are not returning anything but we want to alter the actualy values
void build_layer_sizes(Architecture *arch, int num_inputs, int *out , int *out_len)
{
    out[0] = num_inputs;    //first slot is always for the input layer (decided by that dynamic dataset (the parameter is tied to that(or should be i hope)))
 
    for (int i = 0; i < arch -> num_hidden_layers; i++)     out[i+1] = arch -> layer_sizes[i];   //copy each hidden layer nueron count into the next slot. SO it doesnt interfere with the input layer so instead of {8,16} its {num_inputs , 8 ,16}
    
    out[arch -> num_hidden_layers + 1] = 1; //final slot is always 1 since we are doing binary clasication
    
    *out_len = arch -> num_hidden_layers + 2;   //total length + 2 for the in ands out layers
 
//making sure to return thru the pointer to act make a difference
}
 
 
 
//EVAULATE
//Goal -> takes one nn arc, builds a network from it, trains it, for a short num of epochs(so doesnt take years)(scratch that it still prob will anyway when i increase other things)
//checks it accuracy, stores it in the fitness variable
//free's the network to stop data leek(passaporte)
//this will be called on every arc. MUST FREE MEMEROY OTHERWISE REAL PROBLEM SINCE I LIKE MY COMPUTER
 
//arch -> arc we are trying
//num inputs -> num of input nuerons
//training data -> the full dataset loaded from csv, we train and evaluate on this
//num_rows -0> how many rows are in the dataset
//target col -> the index of the column we are trying to guess the output
//epcohs 
//lr
 
double evaluate(Architecture *arch, int num_inputs, char training_data[][MAX_COLS][MAX_LEN],
     int num_rows, int target_col, int total_cols, int epochs, double lr)
{
    int sizes[MAX_HIDDEN + 2]; //+2 for the in out layer
    int num_layer;
    build_layer_sizes(arch , num_inputs, sizes , &num_layer); //converin the arc into a falt array our initiate network from v6 can use
 
    //creating a net
    Network net;
    init_network(&net, sizes , num_layer); //initialising it
 
    //train for how every many epochs were passed
    for (int e = 0; e < epochs; e++)
    {
        for (int i = 0; i < num_rows; i++)
        {
            double inputs[num_inputs];
            int idx = 0;
 
            for(int col = 0; col < total_cols; col++)
            {
                if (col != target_col) inputs[idx++] = atof(training_data[i][col]);
            }
 
            forward_pass(&net , inputs);
            backprop(&net , atof(training_data[i][target_col]), lr);    //target value comes from target column
        }
    }
 
    //evaulating the nn perfoamce
 
        //count
    int correct = 0;
    for (int i = 0; i < num_rows; i++)
    {
        //settings num of inputs to dynamically found amount
        double inputs[num_inputs];
        int idx = 0;
 
        for (int col = 0; col < total_cols;  col++)     if (col != target_col)      inputs[idx++] = atof(training_data[i][col]);
        
        //final run thru to get results
        forward_pass(&net, inputs);
 
        //grab teh single output nuerons value
        double output = net.layers[net.num_layers - 1].neurons[0].output;
        int pred = output >= 0.5 ? 1:0;     //complexly -> threshold to binary 0 or 1. put simply -> if the prediction is closer to 1 then we make it one and vias versa
        int true_label = (int)round(atof(training_data[i][target_col])); //gets the real true value from the csv (making sure its a int)#
        correct += (pred == true_label); //if true it outputs 1 and adds 1 to the count else its 0 and adds 0        
    }
 
    free_network(&net); //MUST FREE!!!!!!!!!!!!!!!!!!!!!! this could break me
 
    double fitness = (double)correct / num_rows * 100.0; //converts the count to a percentage correct
    arch -> fitness = fitness;
 
    //and finalllllly return fitness
    return fitness;
}
 
//RANDOM BABY ARCHITECTURE
//Goal -> generate a completely origonal random arch to populate the first gen arch
 
//Every run starts with POP_SIZE of these so the seach begins with a diverse arch
//spread the shapes rather than all starting at the same point
//No parameters pure random generation returns a full arch by value
Architecture random_architecture()
{
    Architecture arch;
    arch.num_hidden_layers = (rand() % MAX_HIDDEN) + 1; //random 1 to max hidden +1 stops it being 0
 
    for (int i = 0; i < arch.num_hidden_layers; i++)        arch.layer_sizes[i] = (rand() % 29) + 4;    //random 4 - 32 nueron per layer  //floor of 4 stops layer being to small to learn anything complex (for perspective harley has 20000000 layers but 1 nueron each with no wieght but a huge bias(against me))
    
    arch.fitness = 0.0; //setting to zero to avoid garbage values
 
    return arch; //retuning teh new arch
}   
 
 
//MUATION
//Goal: takes a suruor arch and randomly tweaks it to produce a child. teh child inhertis the general shpe of the [arent bit with on small change this is how the search explores new arcs withpout starting from scratch each time
//this means after genertations good traits from other arcs stay and increase its accuracy
 
//arch - taken BY VALUE not by pointer so we get a copy to modify. and try make a monster
Architecture mutate(Architecture arch)
{
    int mutation = rand() % 3;  //randomly pick a mutation type (0: add layer 1: remove layer 2: nudge nuerons)
    
    if (mutation == 0 && arch.num_hidden_layers < MAX_HIDDEN)
    {
        //add a new hidden layer at the end
        arch.layer_sizes[arch.num_hidden_layers] = (rand() % 29) + 4; //random nueron count for the new layer (but at least more than 4 so it can act learn)
        arch.num_hidden_layers++; //increase the num of layers so its actually used
    }
    else if (mutation == 1 && arch.num_hidden_layers > 1)
    {
        //remove the last layer by js reducing the count of layers
        //the data is still sitting in teh array but num_hidden_layers controls gow many slots are actually used so it gets ingnored
        arch.num_hidden_layers--;
    }
    else
    {
        //nudge a random existing layer neuron count slightly
        int target = rand() % arch.num_hidden_layers;   //pick a random hidden layer to modify
        int change = (rand() % 9) - 4;  //randomly change from - 4 to + 4
 
        arch.layer_sizes[target] += change;     //apply the change 
 
        if (arch.layer_sizes[target] < 2)       arch.layer_sizes[target] = 2;       //floor of 2 stops a layer shrinking by its self to 0 or lower which would break teh network
    }
    arch.fitness = 0.0; //reset fitness to 0 this is now a different untested arc
    return arch;    //return the mutated child for testing
}
 
 
 
 
//MAIN
//Goal ties everything together
//loads the dataset runs the evo search to find the best arc
//then does a fill retrasin on the winner and prints the final results
int main()
{
    //----laod data---
    //load csv
    FILE *file = fopen(file_name, "r");
    if (file == NULL)
    {
        printf("Error opening file\n");
        return 1;
    }
 
    //pre scan header to count columns
    int total_cols = 1; //start with 1 since n commas = n+1 cols
    char header[MAX_HEADER];
    fgets(header, sizeof(header), file); //reads the first lone of the file.
    for (int i = 0; header[i] != '\0'; i++)//goes over the first column untill there istn
    {
        if (header[i] == ',')       total_cols++;   //if there is a , (CSV ONLY) then we add to the count
    }
 
    //asks the USR which index the taget is
    int target_col = 0;
    printf("Enter target column number (1-indexed): ");
    if (scanf("%d", &target_col) != 1)
    {
        printf("Invalid input!\n");
        return 1;
    }
    target_col -= 1;    //convert to 0 based indexing
 
    srand(time(NULL)); //seed randomizer with current time so each run produces origonal output
 
    int num_inputs = total_cols - 1; //-1 since target col
 
    //definging some important varibales for later
    char training_data[MAX_ROWS][MAX_COLS][MAX_LEN];
    char line[MAX_HEADER];   //bad practise but sue me
    int num_rows = 0;
    
    while (fgets(line, sizeof(line), file) && num_rows < MAX_ROWS)
    {
        line[strcspn(line, "\n")] = 0;
        int col = 0;
        char *token = strtok(line, ",");
        while (token != NULL && col < MAX_COLS)
        {
            strncpy(training_data[num_rows][col], token, MAX_LEN - 1);
            training_data[num_rows][col][MAX_LEN - 1] = '\0';//i think im going slgighyu inane
            token = strtok(NULL, ",");
            col++;
        }
        num_rows++;
    }
    fclose(file);
    //I HATE ANJFGNHASJFDN FILE IO
 
 
    //EVO SEARCH
    #define POP_SIZE 10 //how many arc exist per gen
    #define GENERATIONS 10 //how many generations there are
    #define SEARCH_EPOCHS 5000 //short epochs count for each search (if u have a quantum computer)
    #define FULL_EPOCHS 50000 //full epoch count for the final retrain of the sucseeder
 
    //array holding current candidate arcs
    Architecture population[POP_SIZE];
 
 
    //GENERATE INITIAL GENERATION
    //we start with pop_size completely random archs
    //this gives the search a diverse stratging point rather than all candidates beging with the same shape
    for (int i = 0; i < POP_SIZE; i++)
    {
        population[i] = random_architecture();
    }
 
    Architecture best;  //stores the current best arch by tracking it across all gens
    best.fitness = 0.0; //rid garbage values
 
    //MAIN EVO LOOP
    for (int gen = 0; gen < GENERATIONS; gen++)
    {
        printf("\n--- Generation %d ---\n", gen+1); //plus 1 for indexing
 
        //evaulate every arch in the current population
        for (int i = 0; i < POP_SIZE; i++)
        {
            evaluate(&population[i], num_inputs , training_data , num_rows , target_col , total_cols , SEARCH_EPOCHS , 0.1);
            printf("Architecture %d | Hidden layers: %d | Fitness: %.1f%%\n", i, population[i].num_hidden_layers, population[i].fitness);
        
            //check to see if this iteration is the best
            if (population[i].fitness > best.fitness)       best = population[i];//copy the whole thing into the best varaible
        }
 
        //sort population by fitness
        //bubble sort works 
        for (int i = 0; i < POP_SIZE - 1; i++)  for (int j = 0; j < POP_SIZE - i - 1; j++)      if (population[j].fitness < population[j + 1].fitness)
        {
            Architecture temp = population[j];  //swap the two archs
            population[j] = population[j + 1];
            population[j+1] = temp;
        }
        //top half (indexes 0 to 4 survive bottom half are mutated surviors)
        //i % (POP_SIZE / 2) cycles through survivors so each one gets mutated once
        for (int i = POP_SIZE / 2; i < POP_SIZE; i++)       population[i] = mutate(population[i % (POP_SIZE / 2)]);
    }
 
    //RETRAIN WINNER
 
    printf("\n BEST ARCH FOUND | HIDDEN LAYERS : %d | SEARCH FITNESS : %.3f%%\n",best.num_hidden_layers, best.fitness);
    for (int i = 0; i < best.num_hidden_layers; i++)
    {
        printf(" HIDDEN LAYER : %d NUERONS : %d\n", i+1, best.layer_sizes[i]);
    }
    printf("RERTAINING BEST ARCH FOR %d EPOCHS\n", FULL_EPOCHS);
 
    int sizes[MAX_HIDDEN + 2];
    int num_layers;
    build_layer_sizes(&best , num_inputs , sizes , &num_layers); //converts to a usable arch shape
 
 
    //FROM HERE ALMOST IDENTICAL FROM V6
    Network net;
    init_network(&net, sizes, num_layers);
 
    for (int e = 0; e < FULL_EPOCHS; e++)
        for (int i = 0; i < num_rows; i++)
        {
            double inputs[num_inputs];
            int idx = 0;
            for (int col = 0; col < total_cols; col++)
                if (col != target_col) inputs[idx++] = atof(training_data[i][col]);
            forward_pass(&net, inputs);
            backprop(&net, atof(training_data[i][target_col]), 0.1);
        }
 
    //evaluate/print results
    int correct = 0;
    double mse = 0.0;
 
    for (int i = 0; i < num_rows; i++)
    {
        double inputs[num_inputs];
        int idx = 0;
        for (int col = 0; col < total_cols; col++)
            if (col != target_col) inputs[idx++] = atof(training_data[i][col]);
 
        forward_pass(&net, inputs);
        double output = net.layers[net.num_layers - 1].neurons[0].output;
        int pred = output >= 0.5 ? 1 : 0;
        double true_val = atof(training_data[i][target_col]);
        int true_label = (int)round(true_val);
        correct += (pred == true_label);
        mse += (output - true_val) * (output - true_val);
 
        printf("Inputs: (");
        for (int j = 0; j < num_inputs; j++)
        {
            printf("%.3f", inputs[j]);
            if (j < num_inputs - 1) printf(", ");
        }
        printf(") --- Pred: %d --- Raw: %.5f\n", pred, output);
    }
 
    mse /= num_rows;
    printf("\nFinal Accuracy: %.1f%% --- MSE: %.4f\n", (double)correct / num_rows * 100, mse);
 
    free_network(&net);
    return 0;
}
 