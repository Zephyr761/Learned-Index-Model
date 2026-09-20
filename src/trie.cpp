#include "trie.h"
#include <exception>

bool TrieIndex::insert(std::string& key)
{
    int cur = 0;
    TrieNode tmp = this->root;
    while(cur < key.size())
    {
        int i = 0;
        for(i = 0; i < tmp.child.size(); i++)
        {
            if(tmp.child[i].value == key[cur])
            {
                if(cur == key.size() - 1)
                {
                    throw std::runtime_error("repeat key is not allowed");
                    return 0;
                }
                break;
            }
        }
        if(i == tmp.child.size())
        {
            if(tmp.child.size() < MAX)
            {
                for(int j = cur; j < key.size(); j++)
                {
                    TrieNode newNode(key[cur]);
                    tmp.subtree_size++;
                    tmp.child.push_back(newNode);
                    tmp = newNode;
                }
                tmp.terminal = true;
                break;
            }
            else
            {
                throw std::runtime_error("the node is full");
                return 0;
            }
        }
    }
    return 1;
}

bool TrieIndex::contains(const std::string& key)
{
    int cur = 0;
    TrieNode tmp = this->root;
    while(cur < key.size())
    {
        int i = 0;
        for(i = 0; i < tmp.child.size(); i++)
        {
            if(tmp.child[i].value == key[cur])
            {
                break;
            }
        }
        if(i == tmp.child.size()) return 0;
        tmp = tmp.child[i];
        if(cur = key.size() - 1)
        {
            if(tmp.terminal == false) return 0;
            else return 1;
        }
    }
}