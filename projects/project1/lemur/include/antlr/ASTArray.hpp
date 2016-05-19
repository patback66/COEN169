#ifndef INC_ASTArray_hpp__
#define INC_ASTArray_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: ASTArray.hpp,v 1.1 2004/10/08 16:27:34 dfisher Exp $
 */

#include <antlr/config.hpp>
#include <antlr/AST.hpp>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/** ASTArray is a class that allows ANTLR to
  * generate code that can create and initialize an array
  * in one expression, like:
  *    (new ASTArray(3))->add(x)->add(y)->add(z)
  */
class ANTLR_API ASTArray {
public:
	int size; // = 0;
	ANTLR_USE_NAMESPACE(std)vector<RefAST> array;

	ASTArray(int capacity)
	: size(0)
	, array(capacity)
	{
	}

	ASTArray* add(RefAST node)
	{
		array[size++] = node;
		return this;
	}
};

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_ASTArray_hpp__
