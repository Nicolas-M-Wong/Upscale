// C++ program to find adjoint and inverse of a matrix

//Source of the code: 
	//https://www.geeksforgeeks.org/program-to-find-transpose-of-a-matrix/
	//https://www.geeksforgeeks.org/adjoint-inverse-matrix/
//Few addition from the code above : matricial multiplication, transpose.
//Searching to use vector<vector<int>> so that I can use any size of matrix. It is not yet working but hope to make it working soon

#define N 4
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <time.h>
#include <vector>

using namespace std;
using namespace cv;
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
void adjoint(int A[N][N], int adj[N][N])
{
	if (N == 1)
	{
		adj[0][0] = 1;
		return;
	}

	// temp is used to store cofactors of A[][]
	int sign = 1, temp[N][N];

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			// Get cofactor of A[i][j]
			getCofactor(A, temp, i, j, N);

			// sign of adj[j][i] positive if sum of row
			// and column indexes is even.
			sign = ((i + j) % 2 == 0) ? 1 : -1;

			// Interchanging rows and columns to get the
			// transpose of the cofactor matrix
			adj[j][i] = (sign) * (determinant(temp, N - 1));
		}
	}
}

// Function to calculate and store inverse, returns false if
// matrix is singular
bool inverse(int A[N][N], float inverse[N][N])
{
	// Find determinant of A[][]
	int det = determinant(A, N);
	cout << "\nDeterminant :" << det << endl;
	if (det == 0)
	{
		cout << "Singular matrix, can't find its inverse" << endl;
		return false;
	}

	// Find adjoint
	int adj[N][N];
	adjoint(A, adj);

	// Find Inverse using formula "inverse(A) = adj(A)/det(A)"
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			inverse[i][j] = adj[i][j] / float(det);

	return true;
}


void transpose(int A[N][N], int B[N][N])
{
	int i, j;
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			B[i][j] = A[j][i];
	return;
}

void dim_prod_mat(int dim_lineA, int dim_columnB, int dim_line_prod, int dim_column_prod)
{
	dim_line_prod = dim_lineA;
	dim_column_prod = dim_columnB;
	return;
}

void prod_mat(int dim_lineA, int dim_columnA, int dim_lineB, int dim_columnB, int A[N][N], int B[N][N], int prod[N][N])

{
	for (int i = 0; i < dim_lineA; i++)
	{
		for (int j = 0; j < dim_columnB; j++)
		{
			float a_element = 0;
			for (int k = 0; k < dim_lineB; k++)
			{
				a_element = a_element + A[i][k] * B[k][j];
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

void verification_calc()		//Function used to verify is the result of are correct or not
{
	int A[N][N] = { {1, 0, 0, 0},
					{1, 1, 1, 1},
						{0, 1, 0, 0},
					{0, 1, 2, 3} };


	int B[N][N] = { {1, 0, 0, 0},
				{1, 1, 1, 1},
					{0, 1, 0, 0},
				{0, 1, 2, 3} };

	float inv[N][N]; // To store inverse of A[][]

//---------------	Check if we can multiply the two matrix.
	bool status = true;
	int dim_lineB = sizeof(B) / sizeof(B[0]);
	int dim_columnA = sizeof(A[0]) / sizeof(int);
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
		return;
	}
	int prod_[N][N];
	prod_mat(N, N, N, N, A, B, prod_);

	//---------------
	cout << "\nProd of A by B :\n";
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			cout << prod_[i][j] << " ";
		}
		cout << endl;
	}

	cout << "\nThe Inverse is :\n";
	if (inverse(A, inv))
	{
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				cout << inv[i][j] << " ";
			}
			cout << endl;
		}
	}
	return;
}

int main()
{
	time_t start, stop;
	time(&start);
	string name_pic = "C:/Users/Nicolas/Documents/Github_Project/fraise4.jpg";
	Mat image = imread(name_pic);

	if (image.empty())	// Checking if the picture exist
	{					// If the picture doesn't exist then we stop the process and we inform the user of the failure
		cout << "Could not open or find the image" << endl;
		system("pause");
		return -1;
	}

	int axis_1 = image.cols;		//Vertical Resolution of the picture
	int axis_2 = image.rows;		//Horizontal

	// The data is under cv::Mat class
	// image.col(1) gives all the elements present in one column on the three RGB layers
	// image.row(1) same but with the row
	// image.col(1).row(1) give the three layers value for the pixel (1;1)
	Mat image_transpose = image.clone();

	for (int i = 0; i < axis_1; i++)
	{
		for (int j = 0; j < axis_2; j++)
		{
			//auto pixel = pixel_3_layer.at<array<uint8_t, 3>>();
			image_transpose.at<Vec3b>(i, j) = image.at<Vec3b>(j, i);
		}
	}

	Mat black_edge = Mat::zeros(axis_1+2, axis_2+2, CV_8UC3);
	

	imwrite("C:/Users/Nicolas/Documents/Github_Project/chat.jpg", image_transpose);

	/// Black edge 1 pixel wide on each side of the picture ///

	for (int i = 0; i < axis_1; i++)
	{
		for (int j = 0; j < axis_2; j++)
		{
			int new_pos_y = i + 1;
			int new_pos_x = j + 1;
			black_edge.at<Vec3b>(new_pos_x, new_pos_y) = image.at<Vec3b>(j, i);
		}
	}
	cout<<black_edge.col(125).row(2);
	imwrite("C:/Users/Nicolas/Documents/Github_Project/chat2.jpg", black_edge);

	time(&stop);
	unsigned long secondes = (unsigned long)difftime(stop, start);
	cout << "The program took " << setprecision(2) << difftime(stop, start) / 60 << " minutes to complete." << endl;	//Display the time in minutes
	system("pause");
	return 0;
}
