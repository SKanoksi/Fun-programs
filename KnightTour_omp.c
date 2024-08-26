/******************************************************************
*
*  KnightTour_omp.c
*
*  Find closed knight tours, moving around the chess board
*  in parallel (OpenMP threads)
*
*  Copyright (c) 2024, Somrath Kanoksirirath.
*  All rights reserved under BSD 3-clause license.
*
*  Compile: gcc -o kt_omp.exe ./KnightTour_omp.c -fopenmp
*
*  Run: OMP_NUM_THREADS=2 ./kt_omp.exe
*
*  Result: ClosedTour.txt
*
******************************************************************/

// Declare size of board & the wanted number of knight tours ###
// Caution!! LX*LY must not be odd (<-- from Math theorem)
#define LX 6
#define LY 6

#define MAX_START 5            // Maximum number of start position that will run before exit
#define MAX_PATTERN 4          // Maximum number of move pattern to be used per start
#define MAX_TRY 2E8            // Maximum knight move before abort this search (<MAX_MOVE_PATTERN**(LX*LY))
#define RANDOM_MOVE_PATTERN 10 // Shuffle times
#define MAX_MOVE_PATTERN 8

/***************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

typedef int Board_t[LX][LY];
typedef int Track_t[LX*LY][3];
typedef int Move_t[MAX_MOVE_PATTERN][2];

inline int TryMove(const int step, const int k, const Track_t *track, const Board_t *board, const Move_t *move);
inline int* AcceptMove(const int step, const int k, Track_t *track, Board_t *board, const Move_t *move);
inline int* BackTrace(const int step, Track_t *track, Board_t *board);
inline int IsClosedTour(const int x, const int y, const int a, const int b);

inline void ShuffleMovePattern(Move_t *move);
inline void ResetBoard(Board_t *board);
inline void PrintBoard(Board_t *board);
inline void fPrintBoard(FILE *file, Board_t *board);


int main()
{
    srand(time(NULL));

    int num_found = 0 ;
    int num_closed = 0 ;
    int num_opened = 0 ;  // Number of opened tours

    if( MAX_TRY >= pow(MAX_MOVE_PATTERN, LX*LY) ){
        printf("Error:: MAX_TRY is larger than the number of all possible sets of moves. Exit.\n");
        return 0;
    }

    //Save the Result in .txt
    FILE *fc = fopen("ClosedTour.txt","a");

    #pragma omp parallel for
    for(int num_start=0 ; num_start<MAX_START ; ++num_start)
    {
        const int thread_id = omp_get_thread_num();
        Board_t board ;    // Save index of move
        Track_t track ;    // Save track (x, y, currently_searched_move_pattern)
        Move_t move = { {-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1} };

        track[0][0] = 5 ; //rand() % LX ;
        track[0][1] = 5 ; //rand() % LY ;

        for(int num_pattern=0 ; num_pattern<MAX_PATTERN ; ++num_pattern)
        {
            int step = 0 ;
            ShuffleMovePattern(&move);
            ResetBoard(&board);
            board[track[0][0]][track[0][1]] = 1 ;
            track[0][2] = 0 ;

            unsigned long int num_try = 0 ;
            int *k = &track[0][2] ;
            while( num_try<MAX_TRY )
            {
                while( !TryMove(step, *k, &track, &board, &move) )
                {
                    ++num_try;
                    ++*k;
                    while( *k==MAX_MOVE_PATTERN ){
                        k = BackTrace(step--, &track, &board);
                    }
                }

                ++num_try;
                k = AcceptMove(++step, *k, &track, &board, &move);

                if( step == LX*LY-1 )
                {
                    #pragma omp critical
                    {
                        ++num_found;
                        if( IsClosedTour(track[LX*LY-1][0], track[LX*LY-1][1], track[0][0], track[0][1]) )
                        {
                            printf("Thread %d: Found a closed knight tour.\n", thread_id);
                            ++num_closed;
                            //PrintBoard(&board);
                            fPrintBoard(fc, &board);

                        }else{
                            printf("Thread %d: Found an opened knight tour.\n", thread_id);
                            ++num_opened ;
                            //PrintBoard(&board);
                        }
                    }

                    // Backtrace to find next
                    k = BackTrace(step--, &track, &board);
                    while( *k==MAX_MOVE_PATTERN ){
                        k = BackTrace(step--, &track, &board);
                    }

                } // IF :: full board

            } // LOOP :: num_try

        } // LOOP :: num_pattern

    } // LOOP :: num_start

    fclose(fc);

    printf("\nTotally %d tours were found.", num_found);
    printf("\nTotally %d opened tours were found.", num_opened);
    printf("\nTotally %d closed tours were found.\n", num_closed);
    printf("\nNote: Some may be repeated.\n");

return 0; }


// True if not available, False if available
int TryMove(const int step, const int k, const Track_t *track, const Board_t *board, const Move_t *move)
{
    const int x = (*track)[step][0] + (*move)[k][0] ;
    const int y = (*track)[step][1] + (*move)[k][1] ;
    //printf("Step = %d, k = %d -- Try (%d, %d)\n", step, k, x+1, y+1);
    if( (0<=x && x<LX && 0<=y && y<LY ) ){
        return ((*board)[x][y]==0) ;
    }else{
        return 0 ;
    }
}


int* AcceptMove(const int step, const int k, Track_t *track, Board_t *board, const Move_t *move)
{
    (*track)[step][0] = (*track)[step-1][0] + (*move)[k][0] ;
    (*track)[step][1] = (*track)[step-1][1] + (*move)[k][1] ;
    (*track)[step][2] = 0 ;
    (*board)[(*track)[step][0]][(*track)[step][1]] = step+1 ;
    //printf("Step = %d, k = %d -- Accept (%d, %d)\n", step, (*track)[step][2], (*track)[step][0]+1, (*track)[step][1]+1);

return &(*track)[step][2]; }


int* BackTrace(const int step, Track_t *track, Board_t *board)
{
    (*board)[(*track)[step][0]][(*track)[step][1]] = 0 ;
    (*track)[step-1][2] += 1 ;
    //printf("Step = %d, k = %d -- BackTrace (%d,%d)\n", step-1, (*track)[step-1][2] , (*track)[step-1][0]+1, (*track)[step-1][1]+1);

return &(*track)[step-1][2]; }


int IsClosedTour(const int x, const int y, const int a, const int b){
    return ((abs(x-a)==1 && abs(y-b)==2) || (abs(x-a)==2 && abs(y-b)==1)) ;
}


void ShuffleMovePattern(Move_t *move)
{
    for(int i=0 ; i<RANDOM_MOVE_PATTERN ; ++i)
    {
        int a = rand() % MAX_MOVE_PATTERN ;
        int b = rand() % MAX_MOVE_PATTERN ;
        int save[2];

        save[0] = (*move)[a][0];
        save[1] = (*move)[a][1];

        (*move)[a][0] = (*move)[b][0];
        (*move)[a][1] = (*move)[b][1];

        (*move)[b][0] = save[0];
        (*move)[b][1] = save[1];
    }

return; }


void ResetBoard(Board_t *board)
{
    for(int i=0 ; i<LX ; ++i){
        for(int j=0 ; j<LY ; ++j){
            (*board)[i][j] = 0 ;
        }
    }

return; }


void PrintBoard(Board_t *board)
{
    for(int j=LY-1 ; j>=0 ; --j){
        for(int i=0 ; i<LX ; ++i){
            printf(" %d ", (*board)[i][j]);
        }
        printf("\n");
    }

return; }


void fPrintBoard(FILE *file, Board_t *board)
{
    for(int j=LY-1 ; j>=0 ; --j){
        for(int i=0 ; i<LX ; ++i){
            fprintf(file,"%d ", (*board)[i][j]);
        }
        fprintf(file,"\n");
    }
    fprintf(file,"\n");

return; }
