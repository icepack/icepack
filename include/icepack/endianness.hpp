
/**
 * Procedures for converting variables from one endianness to another.
 * Used for reading binary files.
 */

#ifndef ENDIANNESS_HPP
#define ENDIANNESS_HPP

#include <netinet/in.h>

template<class Val> inline Val ntohx(const Val& in)
{
    char out[sizeof(in)] = {0};
    for( size_t i = 0; i < sizeof(Val); ++i )
        out[i] = ((char*)&in)[sizeof(Val)-i-1];
    return *(reinterpret_cast<Val*>(out));
}

template<> inline unsigned char ntohx<unsigned char>(const unsigned char & v )
{
    return v;
}

template<> inline uint16_t ntohx<uint16_t>(const uint16_t & v)
{
    return ntohs(v);
}

template<> inline uint32_t ntohx<uint32_t>(const uint32_t & v)
{
    return ntohl(v);
}

template<> inline uint64_t ntohx<uint64_t>(const uint64_t & v)
{
    uint32_t ret [] =
    {
        ntohl(((const uint32_t*)&v)[1]),
        ntohl(((const uint32_t*)&v)[0])
    };
    return *((uint64_t*)&ret[0]);
}

template<> inline float ntohx<float>(const float& v)
{
    uint32_t const* cast = reinterpret_cast<uint32_t const*>(&v);
    uint32_t ret = ntohx(*cast);
    return *(reinterpret_cast<float*>(&ret));
}

template<class Val> inline Val htonx(const Val& in)
{
    char out[sizeof(in)] = {0};
    for( size_t i = 0; i < sizeof(Val); ++i )
        out[i] = ((char*)&in)[sizeof(Val)-i-1];
    return *(reinterpret_cast<Val*>(out));
}

template<> inline unsigned char htonx<unsigned char>(const unsigned char & v )
{
    return v;
}

template<> inline uint16_t htonx<uint16_t>(const uint16_t & v)
{
    return htons(v);
}

template<> inline uint32_t htonx<uint32_t>(const uint32_t & v)
{
    return htonl(v);
}

template<> inline uint64_t htonx<uint64_t>(const uint64_t & v)
{
    uint32_t ret [] =
    {
        htonl(((const uint32_t*)&v)[1]),
        htonl(((const uint32_t*)&v)[0])
    };
    return *((uint64_t*)&ret[0]);
}

template<> inline float htonx<float>(const float& v)
{
    uint32_t const* cast = reinterpret_cast<uint32_t const*>(&v);
    uint32_t ret = htonx(*cast);
    return *(reinterpret_cast<float*>(&ret));
}

#endif
