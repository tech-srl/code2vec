public boolean skipField(final int tag) throws IOException {
        switch (getTagWireType(tag)) {
        case WireFormat.WIRETYPE_VARINT:
            readInt32();
            return true;
        case WireFormat.WIRETYPE_FIXED64:
            readRawLittleEndian64();
            return true;
        case WireFormat.WIRETYPE_LENGTH_DELIMITED:
            skipRawBytes(readRawVarint32());
            return true;
        case WireFormat.WIRETYPE_START_GROUP:
            skipMessage();
            checkLastTagWas(makeTag(WireFormat.getTagFieldNumber(tag), WireFormat.WIRETYPE_END_GROUP));
            return true;
        case WireFormat.WIRETYPE_END_GROUP:
            return false;
        case WireFormat.WIRETYPE_FIXED32:
            readRawLittleEndian32();
            return true;
        default:
            throw new InvalidProtocolBufferException("Protocol message tag had invalid wire type.");
        }
    }