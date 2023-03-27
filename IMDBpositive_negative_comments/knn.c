#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

double **training_base;
double **test_base;
int confusion_matrix[2][2];
int *votes, *evaluation;
double *distances;
int *distances_index;
int training_size;
int test_size;
int total_features;
double accuracy;
// A utility function to swap two elements
void swap(double* a, double* b)
{
    double t = *a;
    *a = *b;
    *b = t;
}

void swap_index(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}


int partition (double arr[], int low, int high, int arr_index[])
{
    double pivot = arr[high];    // pivot
    int i = (low - 1);  // Index of smaller element
 
    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] <= pivot)
        {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);
            swap_index(&arr_index[i], &arr_index[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    swap_index(&arr_index[i + 1], &arr_index[high]);
    return (i + 1);
}
 
/* The main function that implements QuickSort
 arr[] --> Array to be sorted,
  low  --> Starting index,
  high  --> Ending index */
void quickSort(double arr[], int low, int high, int arr_index[])
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr, low, high, arr_index);
 
        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1,arr_index);
        quickSort(arr, pi + 1, high,arr_index);
    }
}

double manhattan_distance(double *x, double *y){
    double sum = 0.0;
    for(int i = 0; i < total_features - 1; i++){
        sum += fabs(x[i] - y[i]);
    }
    return sum;
}

double euclidian_distance(double *x, double *y){
    double sum = 0.0;
    for(int i = 0; i < total_features - 1; i++){
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }

    return sqrt(sum);
}

double mahalanobis_distance(double *x, double *y){
    //todo
}

double cosine_distance(double *x, double *y){
    //todo
}

double find_min(double *x){
    double xmin = x[0];
    for(int i = 1; i < total_features - 1; i++){
        if(x[i] < xmin){
            xmin = x[i];
        }
    }
    return xmin;
}

double find_max(double *x){
    double xmax = x[0];
    for(int i = 1; i < total_features - 1; i++){
        if(x[i] > xmax){
            xmax = x[i];
        }
    }
    return xmax;
}

void min_max(double *x){
    double xmin = find_min(x);
    double xmax = find_max(x);

    for(int i = 0; i < total_features - 1; i++){
        x[i] = (x[i] - xmin) / (xmax - xmin);
    }
}

double avg(double *x){
    double sum = 0.0;
    for(int i = 0; i < total_features - 1; i++){
        sum += x[i];
    }
    return sum/(double)(total_features - 1);
}

double stdev(double *x){
    double avgX = avg(x);
    double sum = 0.0;

    for(int i = 0; i < total_features - 1; i++){
        sum += ((x[i] - avgX) * (x[i] - avgX));
    }
    return sqrt(sum/(double)(total_features - 1));
}

void z_score(double *x){
    double avgX = avg(x);
    double stdevX = stdev(x);

    printf("\nAverage: %lf\n",avgX);
    printf("\nStandard Deviation: %lf\n",stdevX);

    for(int i = 0; i < total_features - 1; i++){
        x[i] = (x[i] - avgX) / stdevX;
    }
}

void knn(){
    int k,pos = 0,neg = 0,correct=0,incorrect=0;
    printf("\nEnter the value of K (Number of Neighbors): ");
    scanf("%d",&k);

    distances = (double*)malloc(sizeof(double)*training_size);
    evaluation = (int*)malloc(sizeof(int)*test_size);
    distances_index = (int*)malloc(sizeof(int)*training_size);
    int n = training_size;
    for(int i = 0; i < test_size; i++){
        pos = 0;
        neg = 0;
        if(i % 100 == 0)
            printf("%d Samples Classified.\n",i);
        for(int j = 0; j < training_size; j++){
            distances[j] = manhattan_distance(test_base[i],training_base[j]);
            distances_index[j] = j;
        }
        quickSort(distances,0,n-1,distances_index);
        
        for(int j = 0; j < k; j++){
            if((int)training_base[distances_index[j]][total_features-1] == 1)
                pos++;
            else
                neg++;
        }
        if(pos > neg)
            evaluation[i] = 1;
        else
            evaluation[i] = 0;
    }
    printf("Evaluation Finished. Creating Confusion Matrix.\n");
    for(int j = 0; j < test_size; j++){
        //printf("%d",evaluation[j]);
        if((int)test_base[j][total_features-1] == evaluation[j])
            correct++;
        else
            incorrect++;
        confusion_matrix[(int)test_base[j][total_features-1]][evaluation[j]] += 1;
    }
    accuracy = (correct/test_size)*100;
}

void read_file(){
    FILE * fp;
    int count_x = 0;
    int count_y = 0;
    double tmp_feature;
    fp = fopen("reviewsTrainBase.txt", "r");
    
    fscanf(fp,"%d %d",&count_x,&count_y);

    printf("Itens no arquivo: %d\n",count_x);
    printf("Features: %d\n",count_y);

    training_base = (double**)malloc(sizeof(double*)*count_x);
    training_size = count_x;
    total_features = count_y;
    for(int i = 0; i < count_x; i++){
        training_base[i] = (double*)malloc(sizeof(double)*count_y);
    }

    printf("Base Allocated\n");
    /*while ((read = getline(&line, &len, fp)) != -1) {
        printf("Retrieved line of length %zu :\n", read);
        printf("%s", line);
        fscanf(fp, "%s", buff);
    }*/
    
    int i = 0;
    int j = 0;
    while (fscanf(fp,"%lf",&tmp_feature) != EOF) {
        training_base[i][j] = tmp_feature;
        j++;
        if(j == count_y){
            j = 0;
            i++;
        }
    }
    printf("Base Loaded\n");
    fclose(fp);

    fp = fopen("reviewsTestBase.txt", "r");
    
    fscanf(fp,"%d %d",&count_x,&count_y);

    printf("Itens no arquivo: %d\n",count_x);
    printf("Features: %d\n",count_y);

    test_base = (double**)malloc(sizeof(double*)*count_x);
    test_size = count_x;
    for(int i = 0; i < count_x; i++){
        test_base[i] = (double*)malloc(sizeof(double)*count_y);
    }

    printf("Base Allocated\n");
    /*while ((read = getline(&line, &len, fp)) != -1) {
        printf("Retrieved line of length %zu :\n", read);
        printf("%s", line);
        fscanf(fp, "%s", buff);
    }*/
    
    i = 0;
    j = 0;
    while (fscanf(fp,"%lf",&tmp_feature) != EOF) {
        test_base[i][j] = tmp_feature;
        j++;
        if(j == count_y){
            j = 0;
            i++;
        }
    }
    printf("Base Loaded\n");
    fclose(fp);
}

void write_file(){
    printf("Saving knnResult.txt\n");
    FILE * fp;
    fp = fopen("knnResult.txt", "w");

    fprintf(fp,"Accuracy: %lf\n",accuracy);
    fprintf(fp,"Confusion Matrix\n\n");

    fprintf(fp,"\tPos\t\tNeg\n");
    fprintf(fp,"Pos\t%d\t%d\n",confusion_matrix[0][0],confusion_matrix[0][1]);
    fprintf(fp,"Neg\t%d\t%d\n",confusion_matrix[1][0],confusion_matrix[1][1]);

    fclose(fp);

}

int main(void){

    read_file();
    knn();
    write_file();

    return 0;
}