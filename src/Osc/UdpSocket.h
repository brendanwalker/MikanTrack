#pragma once

#include <cstdint>
#include <string>

/// RAII wrapper for a plain unicast UDP sender socket.
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
