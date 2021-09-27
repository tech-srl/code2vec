@Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(getClass().getSimpleName());
        sb.append(" [");
        sb.append("Hash = ").append(hashCode());
        sb.append(", IS_DELETED=").append(IS_DELETED);
        sb.append(", NOT_DELETED=").append(NOT_DELETED);
        sb.append(", id=").append(id);
        sb.append(", title=").append(title);
        sb.append(", subtitle=").append(subtitle);
        sb.append(", price=").append(price);
        sb.append(", readCount=").append(readCount);
        sb.append(", picUrl=").append(picUrl);
        sb.append(", sortOrder=").append(sortOrder);
        sb.append(", goods=").append(goods);
        sb.append(", addTime=").append(addTime);
        sb.append(", updateTime=").append(updateTime);
        sb.append(", deleted=").append(deleted);
        sb.append(", content=").append(content);
        sb.append("]");
        return sb.toString();
    }