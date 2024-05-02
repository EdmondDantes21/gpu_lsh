#include <unordered_map>
#include <unordered_set>
#include <random>
#include "index.hpp"
#include "point.hpp"

using namespace std;

/**
 * @brief Construct a new Index with default values: d = 2 and nbits = 32
 */
Index::Index() {
    this->d = 2;
    this->nbits = 32;
    this->hyperplanes = generate_hyperplanes(2, 32);
}

/**
 * @brief Construct a new Index
 * 
 * @param d Number of dimension
 * @param nbits Number of random hyperplanes
 */
Index::Index(int d, int nbits) {
    this->hyperplanes = generate_hyperplanes(d, nbits);
    this->d = d;
    this->nbits = nbits;
}

Index::~Index() {}

/**
 * @brief Add a point to the index
 */
void Index::add(Point &p) {
    cout << "signature of " << p << " = ";
    vector<bool> sig = signature(p);
    for (int i = 0; i < sig.size(); i++)
        cout << sig[i] << " ";
    cout << endl;
    this->map[signature(p)].insert(p);
}

/**
 * @brief add a vector of points to the index
 * 
 * @param points Is the vector of points to add
*/
void Index::add(vector<Point>& points) {
    // for (int i = 0; i < points.size(); i++)
    //     this->map[signature(points[i])].insert(points[i]);
}

void Index::search(Point& p) {

}

void Index::print() {
    for (auto it = this->map.begin(); it != this->map.end(); it++) {
        cout << "KEY = ";
        for (int i = 0; i < it->first.size(); i++)
            cout << it->first[i] << " ";
        cout << endl;

        int i = 0;
        for (auto &point : it->second) {
            cout << "point " << i++ << endl;
            for (int i = 0; i < point.v.size(); i++) 
                cout << point.v[i] << " ";
            cout << endl;
        }
    }
}

/**
 * @brief Calculates the signature of a point p, i.e. whether it is left/right of each hyperplane defined.
 * 
 * @return The signature of the point p
 */
vector<bool> Index::signature(const Point& p) {
    vector<bool> sig(nbits);
    for (int i = 0; i < nbits; i++) 
        sig[i] = inner_product(p.v.begin(), p.v.end(), this->hyperplanes[i].begin(), 0.0) > 0 ? 1 : 0;
    return sig;
}

/**
 * @brief Wrapper class that contains a method to hash a vector of bools
 */
// struct VectorBoolHash {
//     size_t operator()(const vector<bool>& obj) const {
//         size_t hash = 0;
//         return hash;
//         // return hash<vector<bool>>()(obj);
//     }
// };

/**
 * @brief Wrapper class that contains a method to compare to vectors of bools
 */
struct VectorBoolEqual {
    bool operator()(const vector<bool>& lhs, const vector<bool>& rhs) const {
        return lhs == rhs;
    }
};

/**
 * @brief Generate nbits random hyperplanes
 * 
 * @param d Number of dimensions
 * @param nbits Number of hyperplanes
 * @return Generated hyperplanes
 */
vector<vector<double>> generate_hyperplanes(int d, int nbits) {
    vector<vector<double>> hyperplanes(nbits, vector<double>(d));

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < nbits; i++) 
        for (int j = 0; j < d; j++) 
            hyperplanes[i][j] = dis(gen);
    return hyperplanes;
}