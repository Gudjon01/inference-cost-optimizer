package prefixhash

import (
	"slices"
	"strings"
	"testing"
)

func hashes(h *Hasher, parts ...string) ([]uint64, int) {
	s := h.Start()
	for _, p := range parts {
		s.WriteString(p)
	}
	return s.Hashes(), s.Chars()
}

func TestSharedPrefixSharesLeadingHashes(t *testing.T) {
	h := New([]byte("salt"), 4, 64)
	a, _ := hashes(h, "aaaabbbbccccdddd")
	b, _ := hashes(h, "aaaabbbbXXXXdddd")
	if len(a) != 4 || len(b) != 4 {
		t.Fatalf("want 4 hashes each, got %d and %d", len(a), len(b))
	}
	if !slices.Equal(a[:2], b[:2]) {
		t.Errorf("first two blocks are equal, hashes differ: %x vs %x", a[:2], b[:2])
	}
	// The chain makes every hash after the first difference differ too, even
	// though the last block's own text is the same.
	if a[2] == b[2] || a[3] == b[3] {
		t.Errorf("hashes after the differing block must differ: %x vs %x", a, b)
	}
}

func TestSplitWritesMatchSingleWrite(t *testing.T) {
	h := New([]byte("salt"), 5, 64)
	text := strings.Repeat("şahıs işletmesi ünlü ", 7)
	whole, wholeChars := hashes(h, text)
	for cut := 0; cut <= len(text); cut++ {
		// Writes never split a character in real use, so only cut on boundaries.
		if cut < len(text) && text[cut]&0xC0 == 0x80 {
			continue
		}
		got, chars := hashes(h, text[:cut], text[cut:])
		if !slices.Equal(got, whole) || chars != wholeChars {
			t.Fatalf("cut at %d: got %x (%d chars), want %x (%d chars)", cut, got, chars, whole, wholeChars)
		}
	}
}

func TestBlocksCountCharactersNotBytes(t *testing.T) {
	h := New([]byte("salt"), 4, 64)
	got, chars := hashes(h, "ışğüöçİŞ!") // 9 characters, 17 bytes
	if chars != 9 {
		t.Errorf("chars = %d, want 9", chars)
	}
	if len(got) != 2 {
		t.Errorf("want 2 full blocks of 4 characters, got %d", len(got))
	}
}

func TestPartialBlockIsNotHashed(t *testing.T) {
	h := New([]byte("salt"), 256, 64)
	got, chars := hashes(h, strings.Repeat("x", 255))
	if len(got) != 0 || chars != 255 {
		t.Errorf("got %d hashes over %d chars, want 0 over 255", len(got), chars)
	}
}

func TestMaxBlocksCapsHashesButNotCharCount(t *testing.T) {
	h := New([]byte("salt"), 2, 3)
	got, chars := hashes(h, "aabbccdd", "eeff")
	if len(got) != 3 {
		t.Errorf("got %d hashes, want 3", len(got))
	}
	if chars != 12 {
		t.Errorf("chars = %d, want 12", chars)
	}
}

func TestSaltChangesHashes(t *testing.T) {
	a, _ := hashes(New([]byte("install-a"), 4, 64), "aaaabbbb")
	b, _ := hashes(New([]byte("install-b"), 4, 64), "aaaabbbb")
	if a[0] == b[0] || a[1] == b[1] {
		t.Errorf("different salts gave equal hashes: %x vs %x", a, b)
	}
}

func BenchmarkHash16KChars(b *testing.B) {
	h := New([]byte("0123456789abcdef0123456789abcdef"), 256, 64)
	text := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 400)[:16384]
	b.SetBytes(int64(len(text)))
	for b.Loop() {
		s := h.Start()
		s.WriteString(text)
	}
}
