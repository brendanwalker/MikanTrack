#include "UdpSocket.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
using SockLen= int;
#ifndef SIO_UDP_CONNRESET
#define SIO_UDP_CONNRESET _WSAIOW(IOC_VENDOR, 12)
#endif
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
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

bool UdpSocket::openBound(uint16_t port, bool bBroadcast)
{
	close();

#if defined(_WIN32)
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

	if (bBroadcast)
	{
		const int enable= 1;
		::setsockopt(static_cast<SOCKET>(m_socket), SOL_SOCKET, SO_BROADCAST, reinterpret_cast<const char*>(&enable),
					 sizeof(enable));
	}

#if defined(_WIN32)
	// Without this, an ICMP port-unreachable from an earlier send surfaces as a
	// WSAECONNRESET error on the next receive
	DWORD bytesReturned= 0;
	BOOL bNewBehavior= FALSE;
	::WSAIoctl(static_cast<SOCKET>(m_socket), SIO_UDP_CONNRESET, &bNewBehavior, sizeof(bNewBehavior), nullptr, 0,
			   &bytesReturned, nullptr, nullptr);

	u_long nonBlocking= 1;
	if (::ioctlsocket(static_cast<SOCKET>(m_socket), FIONBIO, &nonBlocking) == SOCKET_ERROR)
	{
		close();
		return false;
	}
#else
	const int flags= ::fcntl(m_socket, F_GETFL, 0);
	if (flags < 0 || ::fcntl(m_socket, F_SETFL, flags | O_NONBLOCK) < 0)
	{
		close();
		return false;
	}
#endif

	sockaddr_in bindAddr{};
	bindAddr.sin_family= AF_INET;
	bindAddr.sin_port= htons(port);
	bindAddr.sin_addr.s_addr= INADDR_ANY;
	if (::bind(static_cast<SOCKET>(m_socket), reinterpret_cast<sockaddr*>(&bindAddr), sizeof(bindAddr)) == SOCKET_ERROR)
	{
		close();
		return false;
	}

	return true;
}

int UdpSocket::receiveFrom(void* buffer, int capacity, std::string* outSourceIP)
{
	if (m_socket == k_invalidSocket)
		return -1;

	for (;;)
	{
		sockaddr_in sourceAddr{};
		SockLen sourceLength= sizeof(sourceAddr);
		int received= ::recvfrom(static_cast<SOCKET>(m_socket), static_cast<char*>(buffer), capacity, 0,
								 reinterpret_cast<sockaddr*>(&sourceAddr), &sourceLength);

#if defined(_WIN32)
		if (received < 0)
		{
			const int error= ::WSAGetLastError();
			if (error == WSAEWOULDBLOCK)
				return 0;
			if (error == WSAECONNRESET)
				continue;
			// Truncated to the buffer: the data that fit is still delivered
			if (error == WSAEMSGSIZE)
				received= capacity;
		}
#else
		if (received < 0)
		{
			if (errno == EWOULDBLOCK || errno == EAGAIN)
				return 0;
			if (errno == EINTR)
				continue;
		}
#endif
		if (received < 0)
			return -1;

		if (outSourceIP != nullptr)
		{
			char text[INET_ADDRSTRLEN]= {};
			::inet_ntop(AF_INET, &sourceAddr.sin_addr, text, sizeof(text));
			*outSourceIP= text;
		}
		return received;
	}
}
