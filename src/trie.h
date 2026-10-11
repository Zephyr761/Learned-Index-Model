#ifndef TRIE_H
#define TRIE_H

#include <iostream>
#include <vector>

#define MAX 64

class TrieNode
{
public:
    bool terminal;
    int subtree_size;
    char value;
    std::vector<TrieNode> child;
    TrieNode(char value)
    {
        this->terminal = 0;
        this->value = value;
        this->subtree_size = 0;
    }
    friend::TrieNode void for_each(const TrieNode& root);
};


int SearchResult(const std::string& key);

class TrieIndex
{
public:
    bool insert(std::string& key);
    bool contains(const std::string& key);
    int find(const std::string& key);
    int lower_bound(const std::string& key);
    int size();
    bool empty();
    bool clear();
    void for_each();
    void range_scan(const std::string& begin_key, const std::string& end_key);
    bool validate();
private:
    TrieNode root;
};
#endif