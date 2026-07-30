#include "UdpSocket.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
using SockLen= int;
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
using SockLen= socklen_t;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define closesocket(s) ::close(s)
#endif

#include <cstring>
#include <cstdio>

UdpSocket::UdpSocket()
	: m_socket(k_invalidSocket)
{
}

UdpSocket::~UdpSocket() { close(); }

bool UdpSocket::open(const std::string& bindIP)
{
	close();

#if defined(_WIN32)
	// WSAStartup/WSACleanup are reference counted by Winsock, so pairing them
	// per-socket-open is safe even if the app initializes Winsock elsewhere.
	WSADATA wsaData{};
	if (::WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
	{
		return false;
	}
	m_wsaInitialized= true;
#endif

	m_socket= static_cast<SocketHandle>(::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP));
	if (m_socket == k_invalidSocket)
	{
		close();
		return false;
	}

	// Bind to the local interface
	sockaddr_in bindAddr{};
	bindAddr.sin_family= AF_INET;
	bindAddr.sin_port= 0; // outgoing, no specific source port needed
	if (bindIP.empty() || bindIP == "0.0.0.0")
	{
		bindAddr.sin_addr.s_addr= INADDR_ANY;
	}
	else
	{
		::inet_pton(AF_INET, bindIP.c_str(), &bindAddr.sin_addr);
	}

	if (::bind(static_cast<SOCKET>(m_socket), reinterpret_cast<sockaddr*>(&bindAddr), sizeof(bindAddr)) == SOCKET_ERROR)
	{
		close();
		return false;
	}

	return true;
}

void UdpSocket::close()
{
	if (m_socket != k_invalidSocket)
	{
		::closesocket(static_cast<SOCKET>(m_socket));
		m_socket= k_invalidSocket;
	}

#if defined(_WIN32)
	if (m_wsaInitialized)
	{
		::WSACleanup();
		m_wsaInitialized= false;
	}
#endif
}

bool UdpSocket::sendTo(const std::string& destIP, uint16_t port, const void* data, int length)
{
	if (m_socket == k_invalidSocket)
		return false;

	sockaddr_in destAddr{};
	destAddr.sin_family= AF_INET;
	destAddr.sin_port= htons(port);
	::inet_pton(AF_INET, destIP.c_str(), &destAddr.sin_addr);

	const int sent= ::sendto(static_cast<SOCKET>(m_socket), static_cast<const char*>(data), length, 0,
							 reinterpret_cast<const sockaddr*>(&destAddr), sizeof(destAddr));

	return sent == length;
}
