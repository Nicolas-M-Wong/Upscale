// C++ program to find adjoint and inverse of a matrix

//Source of the code: 
	//https://www.geeksforgeeks.org/program-to-find-transpose-of-a-matrix/
	//https://www.geeksforgeeks.org/adjoint-inverse-matrix/
//Few addition from the code above : matricial multiplication, transpose.
//Searching to use vector<vector<int>> so that I can use any size of matrix. It is not yet working but hope to make it working soon

#include<bits/stdc++.h>
using namespace std;
#define N 4

// Function to get cofactor of A[p][q] in temp[][]. n is current
// dimension of A[][]
void getCofactor(int A[N][N], int temp[N][N], int p, int q, int n)
{
	int i = 0, j = 0;

	// Looping for each element of the matrix
	for (int row = 0; row < n; row++)
	{
		for (int col = 0; col < n; col++)
		{
			// Copying into temporary matrix only those element
			// which are not in given row and column
			if (row != p && col != q)
			{
				temp[i][j++] = A[row][col];

				// Row is filled, so increase row index and
				// reset col index
				if (j == n - 1)
				{
					j = 0;
					i++;
				}
			}
		}
	}
}

/* Recursive function for finding determinant of matrix.
n is current dimension of A[][]. */
int determinant(int A[N][N], int n)
{
	int D = 0; // Initialize result

	// Base case : if matrix contains single element
	if (n == 1)
		return A[0][0];

	int temp[N][N]; // To store cofactors

	int sign = 1; // To store sign multiplier

	// Iterate for each element of first row
	for (int f = 0; f < n; f++)
	{
		// Getting Cofactor of A[0][f]
		getCofactor(A, temp, 0, f, n);
		D += sign * A[0][f] * determinant(temp, n - 1);

		// terms are to be added with alternate sign
		sign = -sign;
	}

	return D;
}

// Function to get adjoint of A[N][N] in adj[N][N].
void adjoint(int A[N][N],int adj[N][N])
{
	if (N == 1)
	{
		adj[0][0] = 1;
		return;
	}

	// temp is used to store cofactors of A[][]
	int sign = 1, temp[N][N];

	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
		{
			// Get cofactor of A[i][j]
			getCofactor(A, temp, i, j, N);

			// sign of adj[j][i] positive if sum of row
			// and column indexes is even.
			sign = ((i+j)%2==0)? 1: -1;

			// Interchanging rows and columns to get the
			// transpose of the cofactor matrix
			adj[j][i] = (sign)*(determinant(temp, N-1));
		}
	}
}

// Function to calculate and store inverse, returns false if
// matrix is singular
bool inverse(int A[N][N], float inverse[N][N])
{
	// Find determinant of A[][]
	int det = determinant(A, N);
	if (det == 0)
	{
		cout << "Singular matrix, can't find its inverse"<<endl;
		return false;
	}

	// Find adjoint
	int adj[N][N];
	adjoint(A, adj);

	// Find Inverse using formula "inverse(A) = adj(A)/det(A)"
	for (int i=0; i<N; i++)
		for (int j=0; j<N; j++)
			inverse[i][j] = adj[i][j]/float(det);

	return true;
}


void transpose(int A[][N], int B[][N])
{
	int i, j;
	for (i = 0; i < N; i++)
	for (j = 0; j < N; j++)
	    B[i][j] = A[j][i];
	return;
	}

void dim_prod_mat(int dim_lineA,int dim_columnB,int dim_line_prod,int dim_column_prod)
	{
	dim_line_prod = dim_lineA;
	dim_column_prod = dim_columnB;
	return;
	}

void prod_mat (int dim_lineA,int dim_columnA,int dim_lineB,int dim_columnB, int A[N][N], int B[N][N], int prod[N][N])

	{
	for (int i = 0;i<dim_lineA; i++)
		{
		for (int j = 0; j<dim_columnB; j++)
			{
			float a_element = 0;
			for (int k = 0; k<dim_lineB; k++)
				{
				a_element = a_element+A[i][k]*B[k][j];
				}
			prod[i][j] = a_element;
			}
		}
	return;
	}

void error(bool state)
	{
	if (state == false)
		{
		throw "Mismatch dimension !";
		}
	return;
	}

int main()
	{
	int A[N][N] = { {1, 0, 0, 0},
		        {1, 1, 1, 1},
	                {0, 1, 0, 0},
		        {0, 1, 2, 3}};

	int B[N][N] = { {1, 0, 0, 0},
		        {1, 1, 1, 1},
	                {0, 1, 0, 0},
		        {0, 1, 2, 3}};

	transpose(A, B);
	float inv[N][N]; // To store inverse of A[][]

//---------------	Check if we can multiply the two matrix.
	bool status = true;
	int dim_lineB = sizeof(B)/sizeof(B[0]);
	int dim_columnA = sizeof(A[0])/sizeof(int);
	if (dim_columnA != dim_lineB)
		{
		status = false;
		}
	try 
		{
		error(status);
		}
	catch (const char* msg) 
		{
	     	cerr << msg << endl;
		return -1;
		}
	int prod_[N][N];
	prod_mat(N,N,N,N,A,B,prod_);
	
//---------------

	for (int i=0; i<N; i++)
		{
		for (int j=0; j<N; j++)
			{
			cout << prod_[i][j] << " ";
			}
		cout<<endl;
		}

	cout << "\nThe Inverse is :\n";
	if (inverse(A, inv))
		{
			for (int i=0; i<N; i++)
				{
				for (int j=0; j<N; j++)
					{
					cout << inv[i][j] << " ";
					}
				cout<<endl;
			}
		}
	return 0;
}
