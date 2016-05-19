/*
 * @author Matthew Koken <mkoken@scu.edu>
 * COEN 169
 * file: recommendations.cpp
 * Reads in train.txt, a [200] x [1000] tab delimited file of users and movie recs
 * Generates recommendations as results*.txt for test*.txt using cosine similarity
 * and Pearson Correlation
 *
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#define RECS_ROW_USERS 200
#define RECS_COL_MOVIES 1000

int trainRecs[RECS_ROW_USERS][RECS_COL_MOVIES];

using namespace std;

int main(void) {
  //FILE *in = fopen("train.txt", "rt");
  ifstream in;
  in.open("train.txt", ifstream::in);

  if(!in.is_open()) { //check for open error
    cout << "Could not open file!" << endl;
    exit(1);
  } else { //opened correctly, now read it in

    //clear trainrecs
    for(int i = 0; i < RECS_ROW_USERS; i++){
      for(int j = 0; j < RECS_COL_MOVIES; j++)
       trainRecs[i][j] = 0;
    }

    for(int row = 0; row < RECS_ROW_USERS; row++) {
      std::string line;
      std::getline(in, line);

      /*if (!in.good())
         break;*/
      //cout << line << endl;
      std::stringstream iss(line);

      for (int col = 0; col < RECS_COL_MOVIES; col++) {
         std::string val;
         std::getline(iss, val, '\t');
         /*if ( !iss.good() )
             break;*/

         std::stringstream convertor(val);
         //trainRecs[row][col] = atoi(val.c_str());
         convertor >> trainRecs[row][col];
      }
    }

    in.close();
  }

  ofstream outFile("testOut.txt");
  for(int i = 0; i < RECS_ROW_USERS; i++) {
    for(int j = 0; j < RECS_COL_MOVIES; j++) {
      outFile << trainRecs[i][j] << "\t";
    }
    outFile << "\n";
  }
  outFile.close();



  return 0;
}
