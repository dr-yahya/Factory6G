function per = bertoper(ber,PacketSize)
    per = 1-(1-ber).^PacketSize;
end