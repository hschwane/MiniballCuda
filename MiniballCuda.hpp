
//    Modified for use with Cuda: Copright (C) 2020, Hendrik Schwanekamp
//    Original Author: Copright (C) 1999-2013, Bernd Gaertner
//    $Rev: 5891 $
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.

//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.

//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//    Contact:
//    --------
//    Bernd Gaertner
//    Institute of Theoretical Computer Science
//    ETH Zuerich
//    CAB G31.1
//    CH-8092 Zuerich, Switzerland
//    http://www.inf.ethz.ch/personal/gaertner

#include <cassert>
#include <algorithm>
#include <list>
#include <ctime>
#include <limits>

#ifdef __CUDACC__
    #define CUDAHOSTDEV __host__ __device__
#else
    #define CUDAHOSTDEV
#endif

namespace MiniballCuda {

// Helper to mimic std::list in device code
// ================================================

template <typename T, int maxSize> class DeviceListIterator;
template <typename T, int maxSize> class DeviceListConstIterator;

template <typename T, int maxSize>
class DeviceList
{
public:

    CUDAHOSTDEV DeviceList() : m_end(), m_head(&m_end) {}

    typedef DeviceListIterator<T,maxSize> iterator;
    typedef DeviceListConstIterator<T,maxSize> const_iterator;

    CUDAHOSTDEV iterator begin() {return DeviceListIterator<T,maxSize>(m_head);} // iterator to the first element
    CUDAHOSTDEV iterator end() {return DeviceListIterator<T,maxSize>(&m_end);} // iterator to after the last element

    CUDAHOSTDEV const_iterator begin() const {return DeviceListConstIterator<T,maxSize>(m_head);} // iterator to the first element
    CUDAHOSTDEV const_iterator end() const {return DeviceListConstIterator<T,maxSize>(&m_end);} // iterator to after the last element

    // add element in front, adding more than maxSize elements will result in undefined behavior
    CUDAHOSTDEV void push_front(const T v)
    {
        assert(m_size < maxSize);
        m_nodes[m_size].data = v;
        m_nodes[m_size].next = m_head;
        m_nodes[m_size].prev = nullptr;
        m_head->prev = &m_nodes[m_size];
        m_head = &m_nodes[m_size];
        ++m_size;
    }

    // move-add elemnt in front, adding more than maxSize elements will result in undefined behavior
    CUDAHOSTDEV void push_front(T&& v)
    {
        assert(m_size < maxSize);
        m_nodes[m_size].data = std::move(v);
        m_nodes[m_size].next = m_head;
        m_nodes[m_size].prev = nullptr;
        m_head->prev = &m_nodes[m_size];
        m_head = &m_nodes[m_size];
        ++m_size;
    }

    // moves element it to the front of the list
    CUDAHOSTDEV void moveToFront(iterator it)
    {
        Node* node = it.m_node;

        assert(node != &m_end);
        if(node == m_head)
            return;

        Node* oldPrev = node->prev;
        Node* oldNext = node->next;

        // remove node from its old position by closing the gap
        oldPrev->next = oldNext;
        oldNext->prev = oldPrev;

        // now move node to the front of the list
        node->prev = nullptr;
        node->next = m_head;
        m_head->prev = node;
        m_head = node;
    }

    friend class DeviceListIterator<T,maxSize>;
    friend class DeviceListConstIterator<T,maxSize>;

private:

    // struct to store a node
    struct Node
    {
        T data;
        Node* next{nullptr};
        Node* prev{nullptr};
    };

    Node m_nodes[maxSize]; // all list nodes live in this array
    Node m_end; // the end node
    Node* m_head; // current head of the list

    int m_size{0}; // as elements can not be removed the size will only grow
};

template <typename T, int maxSize>
class DeviceListIterator
{
public:
    typedef int difference_type;
    typedef T value_type;
    typedef T& reference;
    typedef T* pointer;
    typedef std::bidirectional_iterator_tag iterator_category;

    // construction
    CUDAHOSTDEV DeviceListIterator() {m_node = nullptr;}
    CUDAHOSTDEV explicit DeviceListIterator(typename DeviceList<T,maxSize>::Node* n) : m_node(n) {};

    // dereferencing
    CUDAHOSTDEV reference operator*() const {return m_node->data;}

    // comparison
    CUDAHOSTDEV bool operator==(const DeviceListIterator& other) const {return (m_node == other.m_node);}
    CUDAHOSTDEV bool operator!=(const DeviceListIterator& other) const {return (m_node != other.m_node);}

    // increment and devrement
    CUDAHOSTDEV DeviceListIterator& operator++()   {assert(m_node); m_node = m_node->next; return *this;}
    CUDAHOSTDEV DeviceListIterator operator++(int) {assert(m_node); DeviceListIterator tmp = *this; m_node = m_node->next; return tmp;}
    CUDAHOSTDEV DeviceListIterator& operator--()   {assert(m_node); m_node = m_node->prev; return *this;}
    CUDAHOSTDEV DeviceListIterator operator--(int) {assert(m_node); DeviceListIterator tmp = *this; m_node = m_node->prev; return tmp;}

    friend class DeviceList<T,maxSize>;
    friend class DeviceListConstIterator<T,maxSize>;
private:
    typename DeviceList<T,maxSize>::Node* m_node;
};

template <typename T, int maxSize>
class DeviceListConstIterator
{
public:
    typedef int difference_type;
    typedef T value_type;
    typedef const T& reference;
    typedef const T* pointer;
    typedef std::bidirectional_iterator_tag iterator_category;

    // construction
    CUDAHOSTDEV DeviceListConstIterator() {m_node = nullptr;}
    CUDAHOSTDEV explicit DeviceListConstIterator(typename DeviceList<T,maxSize>::Node* n) : m_node(n) {};
    CUDAHOSTDEV DeviceListConstIterator( const DeviceListIterator<T,maxSize>& other) : m_node(other.m_node) {};

    // dereferencing
    CUDAHOSTDEV reference operator*() const {return m_node->data;}

    // comparison
    CUDAHOSTDEV bool operator==(const DeviceListConstIterator& other) const {return (m_node == other.m_node);}
    CUDAHOSTDEV bool operator!=(const DeviceListConstIterator& other) const {return (m_node != other.m_node);}

    // increment and devrement
    CUDAHOSTDEV DeviceListConstIterator& operator++()   {assert(m_node); m_node = m_node->next; return *this;}
    CUDAHOSTDEV DeviceListConstIterator operator++(int) {assert(m_node); DeviceListConstIterator tmp = *this; m_node = m_node->next; return tmp;}
    CUDAHOSTDEV DeviceListConstIterator& operator--()   {assert(m_node); m_node = m_node->prev; return *this;}
    CUDAHOSTDEV DeviceListConstIterator operator--(int) {assert(m_node); DeviceListConstIterator tmp = *this; m_node = m_node->prev; return tmp;}

    friend class DeviceList<T,maxSize>;
private:
    typename DeviceList<T,maxSize>::Node* m_node;
};

// Helper to mimic std::find and std::distance in device code
// ================================================

template <typename itT>
CUDAHOSTDEV typename std::iterator_traits<itT>::difference_type deviceDistance(itT first, itT last)
{
    typename std::iterator_traits<itT>::difference_type hops=0;
    while(first != last)
    {
        ++first;
        ++hops;
    }
    return hops;
}

template <typename itT, typename T>
CUDAHOSTDEV itT deviceFind(itT first, itT last, T v)
{
    while(first != last)
    {
        if(*first == v)
            return first;
        ++first;
    }
    return last;
}

// Global Functions
// ================
template <typename NT>
CUDAHOSTDEV inline NT mb_sqr(NT r) { return r * r; }

// Functors
// ========

// functor to map a point iterator to the corresponding coordinate iterator;
// generic version for points whose coordinate containers have begin()
template <typename Pit_, typename Cit_>
struct CoordAccessor
{
    typedef Pit_ Pit;
    typedef Cit_ Cit;

    CUDAHOSTDEV inline Cit operator()(Pit it) const { return (*it).begin(); }
};

// partial specialization for points whose coordinate containers are arrays
template <typename Pit_, typename Cit_>
struct CoordAccessor<Pit_, Cit_*>
{
    typedef Pit_ Pit;
    typedef Cit_* Cit;

    CUDAHOSTDEV inline Cit operator()(Pit it) const { return *it; }
};

// Class Declaration
// =================

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative = true>
class Miniball
{
private:
    // types
    // The iterator type to go through the input points
    typedef typename CoordAccessor::Pit Pit;
    // The iterator type to go through the coordinates of a single point.
    typedef typename CoordAccessor::Cit Cit;
    // The coordinate type
    typedef typename std::iterator_traits<Cit>::value_type NT;
    // The iterator to go through the support points
    typedef typename DeviceList<Pit,maxPoints>::iterator Sit;

    // data members...
    static constexpr int d = dim; // dimension
    Pit points_begin;
    Pit points_end;
    CoordAccessor coord_accessor;
    const NT nt0; // NT(0)

    //...for the algorithms
    DeviceList<Pit,maxPoints> L;
    Sit support_end;
    int fsize;   // number of forced points
    int ssize;   // number of support points

    // ...for the ball updates
    NT* current_c;
    NT current_sqr_r;
    NT c[d + 1][d];
    NT sqr_r[d + 1];

    // helper arrays
    NT q0[d];
    NT z[d + 1];
    NT f[d + 1];
    NT v[d + 1][d];
    NT a[d + 1][d];

    // by how much do we allow points outside?
    NT default_tol;

public:
    // The iterator type to go through the support points
    typedef typename DeviceList<Pit,maxPoints>::const_iterator SupportPointIterator;

    // PRE:  [begin, end) is a nonempty range
    // POST: computes the smallest enclosing ball of the points in the range
    //       [begin, end); the functor a maps a point iterator to an iterator
    //       through the d coordinates of the point
    CUDAHOSTDEV Miniball(Pit begin, Pit end, CoordAccessor ca = CoordAccessor());

    // POST: returns a pointer to the first element of an array that holds
    //       the d coordinates of the center of the computed ball
    CUDAHOSTDEV const NT* center() const;

    // POST: returns the squared radius of the computed ball
    CUDAHOSTDEV NT squared_radius() const;

    // POST: returns the number of support points of the computed ball;
    //       the support points form a minimal set with the same smallest
    //       enclosing ball as the input set; in particular, the support
    //       points are on the boundary of the computed ball, and their
    //       number is at most d+1
    CUDAHOSTDEV int nr_support_points() const;

    // POST: returns an iterator to the first support point
    CUDAHOSTDEV SupportPointIterator support_points_begin() const;

    // POST: returns a past-the-end iterator for the range of support points
    CUDAHOSTDEV SupportPointIterator support_points_end() const;

    // POST: returns the maximum excess of any input point w.r.t. the computed
    //       ball, divided by the squared radius of the computed ball. The
    //       excess of a point is the difference between its squared distance
    //       from the center and the squared radius; Ideally, the return value
    //       is 0. subopt is set to the absolute value of the most negative
    //       coefficient in the affine combination of the support points that
    //       yields the center. Ideally, this is a convex combination, and there
    //       is no negative coefficient in which case subopt is set to 0.
    CUDAHOSTDEV NT relative_error(NT& subopt) const;

    // POST: return true if the relative error is at most tol, and the
    //       suboptimality is 0; the default tolerance is 10 times the
    //       coordinate type's machine epsilon
    CUDAHOSTDEV bool is_valid() const;

private:
    CUDAHOSTDEV void mtf_mb(Sit n);
    CUDAHOSTDEV void mtf_mb_iterative(Sit n); // iterative implementation
    CUDAHOSTDEV void mtf_mb_recursive(Sit n); // original implementation
    CUDAHOSTDEV void mtf_move_to_front(Sit j);
    CUDAHOSTDEV void pivot_mb(Pit n);
    CUDAHOSTDEV void pivot_move_to_front(Pit j);
    CUDAHOSTDEV NT excess(Pit pit) const;
    CUDAHOSTDEV void pop();
    CUDAHOSTDEV bool push(Pit pit);
    CUDAHOSTDEV NT suboptimality() const;
};

// Class Definition
// ================
template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV Miniball<CoordAccessor, dim, maxPoints, useIterative>::Miniball(Pit begin, Pit end,
                                       CoordAccessor ca)
        : points_begin(begin),
          points_end(end),
          coord_accessor(ca),
          nt0(NT(0)),
          L(),
          support_end(L.begin()),
          fsize(0),
          ssize(0),
          current_c(NULL),
          current_sqr_r(NT(-1)),
          default_tol(NT(10) * std::numeric_limits<NT>::epsilon())
{
    assert(points_begin != points_end);

    // set initial center
    for(int j = 0; j < d; ++j)
        c[0][j] = nt0;
    current_c = c[0];

    // compute miniball
    pivot_mb(points_end);
}


template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV const typename Miniball<CoordAccessor, dim, maxPoints, useIterative>::NT*
Miniball<CoordAccessor, dim, maxPoints, useIterative>::center() const
{
    return current_c;
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV typename Miniball<CoordAccessor, dim, maxPoints, useIterative>::NT
Miniball<CoordAccessor, dim, maxPoints, useIterative>::squared_radius() const
{
    return current_sqr_r;
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
int Miniball<CoordAccessor, dim, maxPoints, useIterative>::nr_support_points() const
{
    assert (ssize < d + 2);
    return ssize;
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV typename Miniball<CoordAccessor, dim, maxPoints, useIterative>::SupportPointIterator
Miniball<CoordAccessor, dim, maxPoints, useIterative>::support_points_begin() const
{
    return L.begin();
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV typename Miniball<CoordAccessor, dim, maxPoints, useIterative>::SupportPointIterator
Miniball<CoordAccessor, dim, maxPoints, useIterative>::support_points_end() const
{
    return support_end;
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV typename Miniball<CoordAccessor, dim, maxPoints, useIterative>::NT
Miniball<CoordAccessor, dim, maxPoints, useIterative>::relative_error(NT& subopt) const
{
    NT e, max_e = nt0;
    // compute maximum absolute excess of support points
    for(SupportPointIterator it = support_points_begin();
        it != support_points_end(); ++it)
    {
        e = excess(*it);
        if(e < nt0)
            e = -e;
        if(e > max_e)
        {
            max_e = e;
        }
    }
    // compute maximum excess of any point
    for(Pit i = points_begin; i != points_end; ++i)
        if((e = excess(i)) > max_e)
            max_e = e;

    subopt = suboptimality();
    assert (current_sqr_r > nt0 || max_e == nt0);
    return (current_sqr_r == nt0 ? nt0 : max_e / current_sqr_r);
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV bool Miniball<CoordAccessor, dim, maxPoints, useIterative>::is_valid() const
{
    NT suboptimality;
    return ((relative_error(suboptimality) <= default_tol) && (suboptimality == 0));
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV void Miniball<CoordAccessor, dim, maxPoints, useIterative>::mtf_mb(Sit n)
{
    // Algorithm 1: mtf_mb (L_{n-1}, B), where L_{n-1} = [L.begin, n)
    // B: the set of forced points, defining the current ball
    // S: the superset of support points computed by the algorithm
    // --------------------------------------------------------------
    // from B. Gaertner, Fast and Robust Smallest Enclosing Balls, ESA 1999,
    // http://www.inf.ethz.ch/personal/gaertner/texts/own_work/esa99_final.pdf

    // this is a compile time constant, so there should be zero overhead
    if( useIterative)
        return mtf_mb_iterative(n);
    else
        return mtf_mb_recursive(n);
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV void Miniball<CoordAccessor, dim, maxPoints, useIterative>::mtf_mb_iterative(Miniball::Sit n)
{
    // Algorithm 1: mtf_mb (L_{n-1}, B), where L_{n-1} = [L.begin, n)
    // B: the set of forced points, defining the current ball
    // S: the superset of support points computed by the algorithm
    // --------------------------------------------------------------
    // from B. Gaertner, Fast and Robust Smallest Enclosing Balls, ESA 1999,
    // http://www.inf.ethz.ch/personal/gaertner/texts/own_work/esa99_final.pdf

    // setup a stack
    Sit stackStore[2 * maxPoints];
    Sit* stack = &stackStore[0];

    //   PRE: B = S
    assert (fsize == ssize);

    support_end = L.begin();
    if((fsize) == d + 1)
        return;

    // start at the beginning of L
    Sit i = L.begin();
    Sit j;

    while(stack != &stackStore[0] ||
          i != n) // even if we just resolved the only recursive call, we need to finish the inner loop
    {                                            // we are only done once i == n wile the stack is empty
        // incremental construction
        while(i != n)
        {
            // INV: (support_end - L.begin() == |S|-|B|)
            assert (deviceDistance(L.begin(), support_end) == ssize - fsize);

            j = i++;
            if(excess(*j) > nt0)
                if(push(*j))
                {          // B := B + p_i
                    // emulate recursive call by putting j on the stack and resetting everything else
                    assert(j != L.end());
                    *(stack++) = j;
                    *(stack++) = n;

                    // emulate begin of the function with n = j
                    n = j; // (L_{i-1}, B + p_i)

                    //   PRE: B = S
                    assert (fsize == ssize);
                    support_end = L.begin();
                    if((fsize) == d + 1)
                        break;

                    i = L.begin();
                }
        }

        // if something is on the stack,
        // restore i, n and j from the stack and finish execution of that function
        if(stack != &stackStore[0])
        {
            n = *(--stack);
            j = *(--stack);
            i = j;
            ++i;
            assert(j != L.end());

            pop();                 // B := B - p_i
            mtf_move_to_front(j); // move j to the beginning of the list
        }
    }
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV void Miniball<CoordAccessor, dim, maxPoints, useIterative>::mtf_mb_recursive(Miniball::Sit n)
{
    // Algorithm 1: mtf_mb (L_{n-1}, B), where L_{n-1} = [L.begin, n)
    // B: the set of forced points, defining the current ball
    // S: the superset of support points computed by the algorithm
    // --------------------------------------------------------------
    // from B. Gaertner, Fast and Robust Smallest Enclosing Balls, ESA 1999,
    // http://www.inf.ethz.ch/personal/gaertner/texts/own_work/esa99_final.pdf

    assert (fsize == ssize);

    support_end = L.begin();
    if((fsize) == d + 1)
        return;

    // incremental construction
    for(Sit i = L.begin(); i != n;)
    {
        // INV: (support_end - L.begin() == |S|-|B|)
        assert (deviceDistance(L.begin(), support_end) == ssize - fsize);

        Sit j = i++;
        assert(i != L.end() || n == i );
        if(excess(*j) > nt0)
            if(push(*j))
            {          // B := B + p_i
                assert(j != L.end());
                mtf_mb(j);            // mtf_mb (L_{i-1}, B + p_i)
                pop();                 // B := B - p_i
                mtf_move_to_front(j);
            }
        assert(i != L.end() || n == i );
    }
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV void Miniball<CoordAccessor, dim, maxPoints, useIterative>::mtf_move_to_front(Sit j)
{
    if(support_end == j)
        support_end++;
    L.moveToFront(j);
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV void Miniball<CoordAccessor, dim, maxPoints, useIterative>::pivot_mb(Pit n)
{
    // Algorithm 2: pivot_mb (L_{n-1}), where L_{n-1} = [L.begin, n)
    // --------------------------------------------------------------
    // from B. Gaertner, Fast and Robust Smallest Enclosing Balls, ESA 1999,
    // http://www.inf.ethz.ch/personal/gaertner/texts/own_work/esa99_final.pdf
    NT old_sqr_r;
    const NT* c;
    Pit pivot, k;
    NT e, max_e, sqr_r;
    Cit p;

    int watchdog=0;
    Pit watchedElement = points_end;
    unsigned int loops_without_progress = 0;
    do
    {
        old_sqr_r = current_sqr_r;
        sqr_r = current_sqr_r;

        pivot = points_begin;
        max_e = nt0;
        for(k = points_begin; k != n; ++k)
        {
            p = coord_accessor(k);
            e = -sqr_r;
            c = current_c;
            for(int j = 0; j < d; ++j)
                e += mb_sqr<NT>(*p++ - *c++);
            if(e > max_e)
            {
                max_e = e;
                pivot = k;
            }
        }

        if(sqr_r < 0 || max_e > sqr_r * default_tol)
        {
            // check if the pivot is already contained in the support set
            if(deviceFind(L.begin(), support_end, pivot) == support_end)
            {
                assert (fsize == 0);
                if(push(pivot))
                {
                    mtf_mb(support_end);
                    pop();
                    pivot_move_to_front(pivot);
                }
            }
        }
        if(old_sqr_r < current_sqr_r)
            loops_without_progress = 0;
        else
            ++loops_without_progress;

        // sometimes I get an infinite loop here, where a the same pivots are selected again and again.
        // may be an error in my modifications or in the original implementation
        // anyway, this watchdog is a cruel solution to prevent that as I can not find the actual error
        watchdog++;
        if(watchdog > 10*maxPoints)
        {
            pivot_move_to_front(watchedElement--);
            watchdog = 0;
        }

    } while(loops_without_progress < 2);
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV void Miniball<CoordAccessor, dim, maxPoints, useIterative>::pivot_move_to_front(Pit j)
{
    auto pos = deviceFind(support_end, L.end(), j);
    if( pos == L.end())
        L.push_front(j);
    else
        L.moveToFront(pos);
    if(deviceDistance(L.begin(), support_end) == d + 2)
        support_end--;
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV inline typename Miniball<CoordAccessor, dim, maxPoints, useIterative>::NT
Miniball<CoordAccessor, dim, maxPoints, useIterative>::excess(Pit pit) const
{
    Cit p = coord_accessor(pit);
    NT e = -current_sqr_r;
    NT* c = current_c;
    for(int k = 0; k < d; ++k)
    {
        e += mb_sqr<NT>(*p++ - *c++);
    }
    return e;
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV void Miniball<CoordAccessor, dim, maxPoints, useIterative>::pop()
{
    --fsize;
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV bool Miniball<CoordAccessor, dim, maxPoints, useIterative>::push(Pit pit)
{
    int i, j;
    NT eps = mb_sqr<NT>(std::numeric_limits<NT>::epsilon());

    Cit cit = coord_accessor(pit);
    Cit p = cit;

    if(fsize == 0)
    {
        for(i = 0; i < d; ++i)
            q0[i] = *p++;
        for(i = 0; i < d; ++i)
            c[0][i] = q0[i];
        sqr_r[0] = nt0;
    } else
    {
        // set v_fsize to Q_fsize
        for(i = 0; i < d; ++i)
            //v[fsize][i] = p[i]-q0[i];
            v[fsize][i] = *p++ - q0[i];

        // compute the a_{fsize,i}, i< fsize
        for(i = 1; i < fsize; ++i)
        {
            a[fsize][i] = nt0;
            for(j = 0; j < d; ++j)
                a[fsize][i] += v[i][j] * v[fsize][j];
            a[fsize][i] *= (2 / z[i]);
        }

        // update v_fsize to Q_fsize-\bar{Q}_fsize
        for(i = 1; i < fsize; ++i)
        {
            for(j = 0; j < d; ++j)
                v[fsize][j] -= a[fsize][i] * v[i][j];
        }

        // compute z_fsize
        z[fsize] = nt0;
        for(j = 0; j < d; ++j)
            z[fsize] += mb_sqr<NT>(v[fsize][j]);
        z[fsize] *= 2;

        // reject push if z_fsize too small
        if(z[fsize] < eps * current_sqr_r)
        {
            return false;
        }

        // update c, sqr_r
        p = cit;
        NT e = -sqr_r[fsize - 1];
        for(i = 0; i < d; ++i)
            e += mb_sqr<NT>(*p++ - c[fsize - 1][i]);
        f[fsize] = e / z[fsize];

        for(i = 0; i < d; ++i)
            c[fsize][i] = c[fsize - 1][i] + f[fsize] * v[fsize][i];
        sqr_r[fsize] = sqr_r[fsize - 1] + e * f[fsize] / 2;
    }
    current_c = c[fsize];
    current_sqr_r = sqr_r[fsize];
    ssize = ++fsize;
    return true;
}

template <typename CoordAccessor, int dim, int maxPoints, bool useIterative>
CUDAHOSTDEV typename Miniball<CoordAccessor, dim, maxPoints, useIterative>::NT
Miniball<CoordAccessor, dim, maxPoints, useIterative>::suboptimality() const
{
    NT l[d + 1];
    NT min_l = nt0;
    l[0] = NT(1);
    for(int i = ssize - 1; i > 0; --i)
    {
        l[i] = f[i];
        for(int k = ssize - 1; k > i; --k)
            l[i] -= a[k][i] * l[k];
        if(l[i] < min_l)
            min_l = l[i];
        l[0] -= l[i];
    }
    if(l[0] < min_l)
        min_l = l[0];
    if(min_l < nt0)
        return -min_l;
    return nt0;
}

} // end Namespace Miniball