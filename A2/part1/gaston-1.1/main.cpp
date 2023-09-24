// main.cpp
// Siegfried Nijssen, snijssen@liacs.nl, jan 2004.
#include <iostream>
#include <fstream>
#include "database.h"
#include "path.h"
#include "misc.h"
#include "graphstate.h"
#include <time.h>
#include <unistd.h>
// #include <stdio.h>

using namespace std;

Frequency minfreq = 1;
Database database;
Statistics statistics;
bool dooutput = false;
int phase = 3;
int maxsize = ( 1 << ( sizeof(NodeId)*8 ) ) - 1; // safe default for the largest allowed pattern
FILE *output;

void Statistics::print () {
  int total = 0, total2 = 0, total3 = 0;
  for ( int i = 0; i < frequenttreenumbers.size (); i++ ) {
    cout << "Frequent " << i + 2
         << " cyclic graphs: " << frequentgraphnumbers[i]
         << " real trees: " << frequenttreenumbers[i]
         << " paths: " << frequentpathnumbers[i]
         << " total: " << frequentgraphnumbers[i] + frequenttreenumbers[i] + frequentpathnumbers[i] << endl;
    total += frequentgraphnumbers[i];
    total2 += frequenttreenumbers[i];
    total3 += frequentpathnumbers[i];
  }
  cout << "TOTAL:" << endl
       << "Frequent cyclic graphs: " << total << " real trees: " << total2 << " paths: " << total3 << " total: " << total + total2 + total3 << endl;
}

void puti ( FILE *f, int i ) {
  char array[100];
  int k = 0;
  do {
    array[k] = ( i % 10 ) + '0';
    i /= 10;
    k++;
  }
  while ( i != 0 );
  do {
    k--;
    putc ( array[k], f );
  } while ( k );
}

int main ( int argc, char *argv[] ) {
  clock_t t1 = clock ();
  
  char opt;
  while ( ( opt = getopt ( argc, argv, "m:tp" ) ) != -1 ) {
    switch ( opt ) {
      case 'm': maxsize = atoi ( optarg ) - 1; break;
      case 't': phase = 2; break;
      case 'p': phase = 1; break;
    }
  }
    
  if ( argc - optind < 2 || argc - optind > 3 ) {
    cerr << "Parameters: [-m size] [-p] [-t] support input [output]" << endl;
    return 1;
  }
  
  minfreq = atoi ( argv[optind] );
  // cerr << "Read" << endl;
  FILE *input = fopen ( argv[optind+1], "r" );
  if ( argc - optind == 3 ) {
    dooutput = true;
    output = fopen ( argv[optind+2], "w" );
  }
  database.read ( input );
  fclose ( input );
  // cerr << "Edgecount" << endl;
  database.edgecount ();
  // cerr << "Reorder" << endl;
  database.reorder ();

  initLegStatics ();
  graphstate.init ();
  for ( int i = 0; i < database.nodelabels.size (); i++ ) {
    if ( database.nodelabels[i].frequency >= minfreq &&
         database.nodelabels[i].frequentedgelabels.size () ) {
      Path path ( i );
      path.expand ();
    }
  }

  clock_t t2 = clock ();

  statistics.print ();
  cout << "Approximate total runtime: " << ( (float) t2 - t1 ) / CLOCKS_PER_SEC << "s" << endl;
  if ( argc - optind == 3 )
    fclose ( output );
}
