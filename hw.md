# CS 244 HW 3
## Problem 1
All these options are possible, there is no guaranteed order of execution for pthread. Any thread can finish at any time.
## Problem 2
![code](https://cdn.discordapp.com/attachments/354436683510579202/1237892693707985046/image.png?ex=663d4ce9&is=663bfb69&hm=325ea0c4e28588cd89770e2a6d874bb3d0bc7eda632cf44f24db67024263343c&)
Here is the source code:
~~~#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 5
pthread_t threads[NUM_THREADS];
pthread_mutex_t mutexsum;
int a[2500];
int sum = 0;

void *call_sum(void* tid) {
    int mysum;
    int i = 0;
    int start, end;
    
    int count = tid;
    start = count * 500;

    end =  count * 500 + 499;

    printf("Thread # %d with start = %d and end = %d\n",count,start,end);
    for (i = start; i <= end; i++){
        mysum = mysum + a[i];
    }
    pthread_mutex_lock(&mutexsum);
    sum = sum + mysum;
    pthread_mutex_unlock(&mutexsum);
}


void main(int argv, char* argc)
{
    int i = 0; int rc;
    
    pthread_attr_t attr;
    pthread_mutex_init(&mutexsum, NULL);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_mutex_init(&mutexsum, NULL);
    
    printf("Initializing array : \n");
    
    // set all element = 1 
    for(i=0;i<2500;i++) {
        a[i]=1;
    }
    
    for (i = 0; i < NUM_THREADS; i++) {
        printf("Creating thread # %d.\n", i);
        rc = pthread_create(&threads[i], &attr, &call_sum, (void*)i);
        if (rc) {
            printf("Error in thread %d with rc = %d. \n", i,rc);
            exit(-1);
        }
    }
    
    pthread_attr_destroy(&attr);
    printf("Creating threads complete. start run " );
    
    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("\n\tSum : %d", sum);
    pthread_mutex_destroy(&mutexsum);
    pthread_exit(NULL);

}
~~~

