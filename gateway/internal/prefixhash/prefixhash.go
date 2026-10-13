// Package prefixhash turns request text into a short list of salted hashes
// that reveal which requests share a prefix without keeping any of the text.
//
// The text is cut into fixed blocks of characters. Hash i covers blocks 0..i:
// each hash is chained from the one before it, so two requests have the same
// first n hashes exactly when their first n blocks are identical. The chain
// is an HMAC keyed with a per-install salt, so hashes cannot be compared
// across installs or tested against a dictionary without the salt.
package prefixhash

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"hash"
	"unsafe"
)

type Hasher struct {
	salt       []byte
	blockChars int
	maxBlocks  int
}

func New(salt []byte, blockChars, maxBlocks int) *Hasher {
	return &Hasher{salt: salt, blockChars: blockChars, maxBlocks: maxBlocks}
}

// State hashes one request. Feed it the request text in order with
// WriteString, then read Hashes and Chars.
type State struct {
	h       *Hasher
	mac     hash.Hash
	prev    [sha256.Size]byte
	inBlock int
	total   int
	out     []uint64
}

func (h *Hasher) Start() *State {
	s := &State{h: h, mac: hmac.New(sha256.New, h.salt)}
	s.mac.Write(s.prev[:])
	return s
}

// Full reports whether more text can no longer change the hashes.
func (s *State) Full() bool { return len(s.out) >= s.h.maxBlocks }

func (s *State) WriteString(p string) {
	for len(p) > 0 && !s.Full() {
		n, chars := advance(p, s.h.blockChars-s.inBlock)
		// The hash only reads the bytes, so the string's memory can be
		// passed without a copy.
		s.mac.Write(unsafe.Slice(unsafe.StringData(p), n))
		s.inBlock += chars
		s.total += chars
		p = p[n:]
		if s.inBlock == s.h.blockChars {
			s.closeBlock()
		}
	}
	if len(p) > 0 {
		_, chars := advance(p, len(p))
		s.total += chars
	}
}

func (s *State) closeBlock() {
	sum := s.mac.Sum(s.prev[:0])
	copy(s.prev[:], sum)
	s.out = append(s.out, binary.BigEndian.Uint64(s.prev[:8]))
	s.inBlock = 0
	s.mac.Reset()
	s.mac.Write(s.prev[:])
}

// Hashes returns one hash per complete block. A trailing partial block is
// left out: it could not match a longer request that continues the same text.
func (s *State) Hashes() []uint64 { return s.out }

// Chars is the number of characters written, including those past the last
// hashed block.
func (s *State) Chars() int { return s.total }

// advance returns the byte length of the longest prefix of p holding at most
// want characters, and how many characters that prefix holds.
func advance(p string, want int) (n, chars int) {
	for i := 0; i < len(p); i++ {
		if p[i]&0xC0 != 0x80 { // first byte of a character
			if chars == want {
				return i, chars
			}
			chars++
		}
	}
	return len(p), chars
}
