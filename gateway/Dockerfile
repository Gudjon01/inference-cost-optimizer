FROM golang:1.27 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway \
 && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/mockengine ./cmd/mockengine \
 && mkdir /state

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/gateway /out/mockengine /usr/local/bin/
# The hash salt is written here on first start; mount a volume to keep it.
COPY --from=build --chown=nonroot:nonroot /state /var/lib/ico
EXPOSE 8080 8081
ENTRYPOINT ["/usr/local/bin/gateway"]
