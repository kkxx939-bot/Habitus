# 测试专用 TLS 证书（W18）

**只给测试用，不得用于任何真实服务。** 私钥（`ca.key`、`server.key`）随仓库公开，谁都能拿它签证书，所以这套 CA
永远不该进任何系统证书库。

| 文件 | 内容 |
|---|---|
| `ca.pem` / `ca.key` | 自签根证书 `CN=Habitus Test-Only CA`，有效期 100 年 |
| `server.pem` / `server.key` | 由上面 CA 签发的服务端证书，SAN = `DNS:model.habitus.test`、`IP:127.0.0.1`，EKU = serverAuth |
| `ca.cnf`、`server.cnf`、`sign.cnf` | 生成上面证书用的 OpenSSL 配置 |

用法：测试在本机起一个用 `server.pem` 的 HTTPS 服务（地址 `127.0.0.1`），或者让测试代理把
`CONNECT model.habitus.test:443` 接到这个服务；客户端把 `ca.pem` 配成 `network.ca_bundle` 才信任它。

## 为什么服务端证书的起始日期是 2019-01-01

macOS 的系统证书校验（`truststore` 在 macOS 上走 Security.framework）要求 **2019-07-01 之后签发**的 TLS
服务端证书有效期不超过 825 天，否则报 "certificate is not standards compliant"；就算 CA 是通过
`load_verify_locations` 额外加进来的也一样。测试证书要长期有效（100 年），所以把起始日期定在
2019-01-01（这条规则之前），截止 2126 年。Linux（OpenSSL）与 Windows 不受影响。

## 重新生成

在本目录执行（macOS 自带的 LibreSSL 与 OpenSSL 3 都可以）：

```sh
# 1. 根 CA（100 年）
openssl req -x509 -new -newkey rsa:2048 -nodes -keyout ca.key -out ca.pem -days 36500 -sha256 -config ca.cnf

# 2. 服务端私钥与签名请求
openssl req -new -newkey rsa:2048 -nodes -keyout server.key -out server.csr -config server.cnf

# 3. 用 CA 签发，起始日期早于 2019-07-01（原因见上）
mkdir -p .ca-db && : > .ca-db/index.txt && echo 1000 > .ca-db/serial
openssl ca -batch -config sign.cnf -in server.csr -out server.pem \
  -startdate 20190101000000Z -enddate 21261012000000Z \
  -extfile server.cnf -extensions v3_server -notext

# 4. 清理中间文件
rm -rf .ca-db server.csr
```
