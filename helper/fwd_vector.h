#ifndef CORE_FWD_VECTOR_H
#define CORE_FWD_VECTOR_H


namespace std {
    template <typename>  class allocator;
    template <typename, typename> class vector;
}

namespace core {
    template<class T> using fwd_vector = std::vector<T, std::allocator<T> >;
}

#endif //CORE_FWD_VECTOR_H
