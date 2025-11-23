"""
Security module for iris-vector-rag.

This module provides Role-Based Access Control (RBAC) integration for
multi-tenant deployments and enterprise security requirements.

Classes:
    RBACPolicy: Abstract base class for implementing custom RBAC policies
    PermissionDeniedError: Exception raised when user lacks required permissions

Example:
    >>> from iris_vector_rag.security import RBACPolicy
    >>>
    >>> class MyRBACPolicy(RBACPolicy):
    ...     def check_collection_access(self, user, collection_id, operation):
    ...         # Custom authorization logic
    ...         return user.role in ['admin', 'reader']
    ...
    ...     def filter_documents(self, user, documents):
    ...         # Filter documents based on user permissions
    ...         return [doc for doc in documents if self._can_access(user, doc)]
    >>>
    >>> # Use with IRISVectorStore
    >>> policy = MyRBACPolicy()
    >>> store = IRISVectorStore(rbac_policy=policy)
"""

from iris_vector_rag.security.rbac import RBACPolicy
from iris_vector_rag.exceptions import PermissionDeniedError

__all__ = [
    "RBACPolicy",
    "PermissionDeniedError",
]
