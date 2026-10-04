#pragma once

#include <cstdint>
#include <string>

/// RAII wrapper for a UDP socket: a unicast sender, or a bound non-blocking receiver.
/// Windows: Winsock2.  Linux/macOS: POSIX sockets.
class UdpSocket
{
public:
	UdpSocket();
	~UdpSocket();

	// Non-copyable
	UdpSocket(const UdpSocket&)= delete;
	UdpSocket& operator=(const UdpSocket&)= delete;

	/// Open and bind the socket.
	/// @param bindIP  Local interface IP to bind to (empty or "0.0.0.0" for default)
	/// @returns true on success
	bool open(const std::string& bindIP= std::string());

	/// Close the socket.
	void close();

	bool isOpen() const { return m_socket != k_invalidSocket; }

	/// Send a raw buffer to the given destination IP and port (UDP unicast).
	/// @param destIP    Destination IP string (e.g., "127.0.0.1")
	/// @param port      Destination UDP port
	/// @param data      Pointer to data buffer
	/// @param length    Number of bytes to send
	/// @returns true if all bytes were sent
	bool sendTo(const std::string& destIP, uint16_t port, const void* data, int length);

	/// Open the socket bound to a specific local port on every interface, for
	/// receiving. The socket is non-blocking and can also send.
	/// @param port        Local UDP port to bind
	/// @param bBroadcast  Enable SO_BROADCAST so sendTo may target a broadcast address
	/// @returns true on success
	bool openBound(uint16_t port, bool bBroadcast);

	/// Receive one waiting datagram without blocking.
	/// @param buffer       Destination buffer
	/// @param capacity     Size of the buffer; a longer datagram is truncated
	/// @param outSourceIP  Optional, receives the sender's IPv4 address text
	/// @returns bytes received, 0 when nothing is waiting, -1 on error
	int receiveFrom(void* buffer, int capacity, std::string* outSourceIP= nullptr);

private:
#if defined(_WIN32)
	using SocketHandle= uintptr_t;
	static constexpr SocketHandle k_invalidSocket= static_cast<SocketHandle>(~0);

	bool m_wsaInitialized= false;
#else
	using SocketHandle= int;
	static constexpr SocketHandle k_invalidSocket= -1;
#endif

	SocketHandle m_socket= k_invalidSocket;
};
