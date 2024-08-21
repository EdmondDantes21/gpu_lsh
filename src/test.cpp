#include <iostream>
using namespace std;

double corput(int n, int base) {
    double q = 0, bk = (double)1 / base;

    while (n > 0) {
        q += (n % base) * bk;
        n /= base;
        bk /= base;
        cout << "q = " << q << endl;
    }

    return q;
}

int main() {
    int n = 8, b = 3;
    cout << " n = " << n << ", b = " << b << ": " << corput(n, b) << endl;
}